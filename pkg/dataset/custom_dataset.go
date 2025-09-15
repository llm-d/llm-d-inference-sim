/*
Copyright 2025 The llm-d-inference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package dataset

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	_ "github.com/mattn/go-sqlite3"
)

type CustomDataset struct {
	BaseDataset
	db        *sql.DB
	hasWarned bool
}

// use constants for expected column names and types
const (
	tableName         = "llmd"
	promptHashCol     = "prompt_hash"
	genTokensCol      = "gen_tokens"
	nGenTokensCol     = "n_gen_tokens"
	promptHashColType = "BLOB"
	genTokensColType  = "JSON"
	nGenTokensColType = "INTEGER"
)

func (d *CustomDataset) downloadDataset(url string, savePath string) error {
	// Set up signal handling for Ctrl+C (SIGINT)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigs)

	// Goroutine to listen for signal
	go func() {
		<-sigs
		d.Logger.Info("Interrupt signal received, cancelling download...")
		cancel()
	}()

	out, err := os.Create(savePath)
	if err != nil {
		return err
	}
	defer func() {
		cerr := out.Close()
		if cerr != nil {
			d.Logger.Error(cerr, "failed to close file after download")
		}
	}()

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer func() {
		cerr := resp.Body.Close()
		if cerr != nil {
			d.Logger.Error(cerr, "failed to close response body after download")
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Progress reader with context
	pr := &progressReader{
		Reader:        resp.Body,
		total:         resp.ContentLength,
		logger:        d.Logger,
		ctx:           ctx,
		startTime:     time.Now(),
		hasShownSpeed: false,
	}

	written, err := io.Copy(out, pr)
	if err != nil {
		// Remove incomplete file
		cerr := os.Remove(savePath)
		if cerr != nil {
			d.Logger.Error(cerr, "failed to remove incomplete file after download")
		}
		// If context was cancelled, return a specific error
		if errors.Is(err, context.Canceled) {
			return errors.New("download cancelled by user")
		}
		return fmt.Errorf("failed to download file: %w", err)
	}
	// Check if file size is zero or suspiciously small
	if written == 0 {
		cerr := os.Remove(savePath)
		if cerr != nil {
			d.Logger.Error(cerr, "failed to remove empty file after download")
		}
		return errors.New("downloaded file is empty")
	}

	// Ensure file is fully flushed and closed before returning success
	if err := out.Sync(); err != nil {
		cerr := os.Remove(savePath)
		if cerr != nil {
			d.Logger.Error(cerr, "failed to remove incomplete file after download")
		}
		return fmt.Errorf("failed to sync file: %w", err)
	}

	return nil
}

// progressReader wraps an io.Reader and logs download progress.
type progressReader struct {
	io.Reader
	total         int64
	downloaded    int64
	startTime     time.Time
	lastPct       int
	logger        logr.Logger
	ctx           context.Context
	hasShownSpeed bool
}

func (pr *progressReader) Read(p []byte) (int, error) {
	select {
	case <-pr.ctx.Done():
		return 0, pr.ctx.Err()
	default:
	}
	n, err := pr.Reader.Read(p)
	pr.downloaded += int64(n)
	if pr.total > 0 {
		pct := int(float64(pr.downloaded) * 100 / float64(pr.total))
		if !pr.hasShownSpeed && time.Since(pr.startTime).Seconds() > 2 {
			pr.hasShownSpeed = true
			pr.logProgress(pct)
			pr.lastPct = pct
		}
		if pct != pr.lastPct && pct%10 == 0 {
			pr.logProgress(pct)
			pr.lastPct = pct
		}
	}
	return n, err
}

func (pr *progressReader) logProgress(pct int) {
	elapsedTime := time.Since(pr.startTime).Seconds()
	speed := float64(pr.downloaded) / (1024 * 1024 * elapsedTime)
	remainingTime := float64(pr.total-pr.downloaded) / (float64(pr.downloaded) / elapsedTime)
	if pct != 100 {
		pr.logger.Info(fmt.Sprintf("Download progress: %d%%, Speed: %.2f MB/s, Remaining time: %.2fs", pct, speed, remainingTime))
	} else {
		pr.logger.Info(fmt.Sprintf("Download completed: 100%%, Average Speed: %.2f MB/s, Total time: %.2fs", speed, elapsedTime))
	}
}

func (d *CustomDataset) verifyDB() error {
	rows, err := d.db.Query("PRAGMA table_info(" + tableName + ");")
	if err != nil {
		return fmt.Errorf("failed to query table info for `%s`: %w", tableName, err)
	}
	defer func() {
		if cerr := rows.Close(); cerr != nil {
			d.Logger.Error(cerr, "failed to close rows after querying table info")
		}
	}()

	expectedColumns := map[string]string{
		promptHashCol: promptHashColType,
		genTokensCol:  genTokensColType,
		nGenTokensCol: nGenTokensColType,
	}

	columnsFound := make(map[string]bool)

	var (
		columnName string
		columnType string
		cid        int
		notnull    int
		dfltValue  interface{}
		pk         int
	)

	for rows.Next() {
		err := rows.Scan(&cid, &columnName, &columnType, &notnull, &dfltValue, &pk)
		if err != nil {
			return fmt.Errorf("failed to scan table info row: %w", err)
		}
		if expectedType, exists := expectedColumns[columnName]; exists {
			if columnType != expectedType {
				return fmt.Errorf("column %s has incorrect type: expected %s, got %s", columnName, expectedType, columnType)
			}
			columnsFound[columnName] = true
		}
	}

	for col := range expectedColumns {
		if !columnsFound[col] {
			return fmt.Errorf("missing expected column in %s table: %s", tableName, col)
		}
	}

	return nil
}

func (d *CustomDataset) getRecordsCount() (int, error) {
	var count int
	err := d.db.QueryRow("SELECT COUNT(" + promptHashCol + ") FROM " + tableName + ";").Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to query database: %w", err)
	}
	return count, nil
}

func (d *CustomDataset) connectToDB(path string) error {
	if d.db != nil {
		err := d.db.Close()
		if err != nil {
			d.Logger.Error(err, "failed to close existing database connection")
		}
		d.db = nil
	}
	// check if file exists
	_, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("database file does not exist: %w", err)
	}
	d.db, err = sql.Open("sqlite3", path)
	if err != nil {
		return fmt.Errorf("failed to open database: %w", err)
	}

	err = d.verifyDB()

	if err != nil {
		return fmt.Errorf("failed to verify database: %w", err)
	}

	count, err := d.getRecordsCount()
	if err != nil {
		d.Logger.Error(err, "failed to get records count")
		return fmt.Errorf("failed to query database: %w", err)
	}
	d.Logger.Info("Database connected successfully", "path", path, "records count", count)

	return nil
}

func (d *CustomDataset) Init(path string, url string, savePath string) error {
	d.hasWarned = false
	if path != "" {
		return d.connectToDB(path)
	}
	if url != "" {
		if savePath == "" {
			user, err := os.UserHomeDir()
			if err != nil {
				return fmt.Errorf("failed to get user home directory: %w", err)
			}
			savePath = filepath.Join(user, ".llm-d", "dataset.sqlite3")
		}

		_, err := os.Stat(savePath)
		if err != nil {
			// file does not exist, download it
			folder := filepath.Dir(savePath)
			err := os.MkdirAll(folder, 0755)
			if err != nil {
				return fmt.Errorf("failed to create parent directory: %w", err)
			}
			d.Logger.Info("Downloading dataset from URL", "url", url, "to", savePath)
			err = d.downloadDataset(url, savePath)
			if err != nil {
				return fmt.Errorf("failed to download dataset: %w", err)
			}
		}
		d.Logger.Info("Using dataset from", "path", savePath)

		return d.connectToDB(savePath)
	}
	return errors.New("no dataset path or url provided")
}

func (d *CustomDataset) Close() error {
	if d.db != nil {
		return d.db.Close()
	}
	return nil
}

func unmarshalAllRecords(rows *sql.Rows) ([][]string, error) {
	var tokensList [][]string
	for rows.Next() {
		var tokensJSON string
		if err := rows.Scan(&tokensJSON); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		var tokens []string
		if err := json.Unmarshal([]byte(tokensJSON), &tokens); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tokens JSON: %w", err)
		}
		tokensList = append(tokensList, tokens)
	}
	return tokensList, nil
}

func (d *CustomDataset) GetPromptHash(req openaiserverapi.CompletionRequest) []byte {
	hashArray := sha256.Sum256([]byte(req.GetFullPrompt()))
	return hashArray[:]
}

func (d *CustomDataset) GetPromptHashHex(hashBytes []byte) string {
	return hex.EncodeToString(hashBytes)
}

// GetTokens returns tokens and finishReason for the given request and mode (echo or random)
func (d *CustomDataset) GetTokens(req openaiserverapi.CompletionRequest, mode string) ([]string, string, error) {
	if mode == common.ModeEcho {
		return d.echo(req)
	}
	nTokensToGen, finishReason := howManyTokensToGen(d.extractMaxTokens(req), req.GetIgnoreEOS())
	tokens, err := d.GenerateTokens(req, nTokensToGen)
	return tokens, finishReason, err
}

func (d *CustomDataset) GenerateTokens(req openaiserverapi.CompletionRequest, nTokens int) ([]string, error) {
	promptHash := d.GetPromptHash(req)
	promptHashHex := d.GetPromptHashHex(promptHash)
	rows, err := d.db.Query("SELECT " + genTokensCol + " FROM " + tableName + " WHERE " + promptHashCol + "=X'" + promptHashHex + "';")
	if err != nil {
		if !d.hasWarned {
			d.Logger.Error(err, "failed to query database. Ensure the prompt hash exists in the dataset. Will generate random tokens instead.")
			d.hasWarned = true
		}
		return GenPresetRandomTokens(nTokens), nil
	}
	defer func() {
		if cerr := rows.Close(); cerr != nil {
			d.Logger.Error(cerr, "failed to close rows after query")
		}
	}()

	tokensList, err := unmarshalAllRecords(rows)
	if err != nil {
		d.Logger.Error(err, "failed to unmarshal records from database")
		return GenPresetRandomTokens(nTokens), nil
	}

	if len(tokensList) == 0 {
		return GenPresetRandomTokens(nTokens), nil
	}
	d.hasWarned = false
	randIndex := rand.Intn(len(tokensList))
	return tokensList[randIndex], nil
}
