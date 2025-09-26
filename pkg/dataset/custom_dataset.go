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
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
)

type CustomDataset struct {
	BaseDataset
	db        *sql.DB
	hasWarned bool
}

// use constants for expected column names and types
const (
	tableName                  = "llmd"
	idCol                      = "id"
	promptHashCol              = "prompt_hash"
	genTokensCol               = "gen_tokens"
	nGenTokensCol              = "n_gen_tokens"
	idColType                  = "INTEGER"
	promptHashColType          = "BLOB"
	genTokensColType           = "JSON"
	nGenTokensColType          = "INTEGER"
	progressLogTimeInterval    = 5 * time.Second
	progressLogPercentInterval = 10
)

func (d *CustomDataset) downloadDataset(ctx context.Context, url string, path string) error {
	folder := filepath.Dir(path)
	err := os.MkdirAll(folder, 0755)
	if err != nil {
		return fmt.Errorf("failed to create parent directory: %w", err)
	}

	if _, err := os.Stat(path); err == nil {
		// file already exists
		return errors.New("Dataset file already exists, should not download: " + path)
	}

	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		cerr := out.Close()
		if cerr != nil {
			d.logger.Error(cerr, "failed to close file after download")
		}
	}()

	d.logger.Info("Using dataset-url", "dataset-url", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer func() {
		cerr := resp.Body.Close()
		if cerr != nil {
			d.logger.Error(cerr, "failed to close response body after download")
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Progress reader with context
	pr := &progressReader{
		Reader:    resp.Body,
		total:     resp.ContentLength,
		logger:    d.logger,
		ctx:       ctx,
		startTime: time.Now(),
	}

	written, err := io.Copy(out, pr)
	if err != nil {
		// Remove incomplete file
		cerr := os.Remove(path)
		if cerr != nil {
			d.logger.Error(cerr, "failed to remove incomplete file after download")
		}
		// If context was cancelled, return a specific error
		if errors.Is(err, context.Canceled) {
			return errors.New("download cancelled by user")
		}
		return fmt.Errorf("failed to download file: %w", err)
	}
	// Check if file size is zero
	if written == 0 {
		cerr := os.Remove(path)
		if cerr != nil {
			d.logger.Error(cerr, "failed to remove empty file after download")
		}
		return errors.New("downloaded file is empty")
	}

	// Ensure file is fully flushed and closed before returning success
	if err := out.Sync(); err != nil {
		cerr := os.Remove(path)
		if cerr != nil {
			d.logger.Error(cerr, "failed to remove incomplete file after download")
		}
		return fmt.Errorf("failed to sync file: %w", err)
	}

	return nil
}

// progressReader wraps an io.Reader and logs download progress.
type progressReader struct {
	io.Reader
	total       int64
	downloaded  int64
	startTime   time.Time
	lastPct     int
	lastLogTime time.Time
	logger      logr.Logger
	ctx         context.Context
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
		now := time.Now()

		timeSinceLastLog := now.Sub(pr.lastLogTime).Seconds()
		pctDiff := pct - pr.lastPct

		if timeSinceLastLog >= progressLogTimeInterval.Seconds() || (pctDiff >= progressLogPercentInterval && pct != pr.lastPct) {
			// progress will be shown every interval seconds or every interval percent of progress
			pr.logProgress(pct)
			pr.lastPct = pct
			pr.lastLogTime = now
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
			d.logger.Error(cerr, "failed to close rows after querying table info")
		}
	}()

	expectedColumns := map[string]string{
		idCol:         idColType,
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

func (d *CustomDataset) loadDatabaseInMemory(path string) error {
	d.logger.Info("Loading database into memory...")
	start := time.Now()

	// Create in-memory database
	var err error
	d.db, err = sql.Open("sqlite3", ":memory:")
	if err != nil {
		return fmt.Errorf("failed to create in-memory database: %w", err)
	}

	// Use ATTACH to copy the database
	attachSQL := fmt.Sprintf("ATTACH DATABASE '%s' AS source", path)
	_, err = d.db.Exec(attachSQL)
	if err != nil {
		if closeErr := d.db.Close(); closeErr != nil {
			d.logger.Error(closeErr, "failed to close in-memory database after attach failure")
		}
		d.db = nil
		return fmt.Errorf("failed to attach source database: %w", err)
	}

	// Copy the table structure first
	_, err = d.db.Exec(`CREATE TABLE llmd (
		id INTEGER PRIMARY KEY,
		prompt_hash BLOB,
		gen_tokens JSON,
		n_gen_tokens INTEGER
	)`)
	if err != nil {
		if closeErr := d.db.Close(); closeErr != nil {
			d.logger.Error(closeErr, "failed to close in-memory database after create table failure")
		}
		d.db = nil
		return fmt.Errorf("failed to create table: %w", err)
	}

	// Copy the data
	_, err = d.db.Exec("INSERT INTO llmd SELECT * FROM source.llmd")
	if err != nil {
		if closeErr := d.db.Close(); closeErr != nil {
			d.logger.Error(closeErr, "failed to close in-memory database after copy failure")
		}
		d.db = nil
		return fmt.Errorf("failed to copy data: %w", err)
	}

	// Detach the source database
	_, err = d.db.Exec("DETACH DATABASE source")
	if err != nil {
		d.logger.Error(err, "failed to detach source database")
	}

	loadTime := time.Since(start)
	d.logger.Info("Database loaded into memory", "load_time", loadTime.String())
	return nil
}

func (d *CustomDataset) connectToDB(path string, useInMemory bool) error {
	if d.db != nil {
		err := d.db.Close()
		if err != nil {
			d.logger.Error(err, "failed to close existing database connection")
		}
		d.db = nil
	}
	// check if file exists
	_, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("database file does not exist: %w", err)
	}

	if useInMemory {
		err = d.loadDatabaseInMemory(path)
		if err != nil {
			return err
		}
	} else {
		// Use file-based database (original behavior)
		d.db, err = sql.Open("sqlite3", path)
		if err != nil {
			return fmt.Errorf("failed to open database: %w", err)
		}

		// Check if there are other connections to the database
		_, err = d.db.Exec("BEGIN EXCLUSIVE;")
		if err != nil {
			if closeErr := d.db.Close(); closeErr != nil {
				d.logger.Error(closeErr, "failed to close database after failing to acquire exclusive lock")
			}
			d.db = nil
			return fmt.Errorf("database is locked or has other active connections: %w", err)
		}
	}

	err = d.verifyDB()
	if err != nil {
		return fmt.Errorf("failed to verify database: %w", err)
	}

	count, err := d.getRecordsCount()
	if err != nil {
		d.logger.Error(err, "failed to get records count")
		return fmt.Errorf("failed to query database: %w", err)
	}

	if useInMemory {
		d.logger.Info("In-memory database connected successfully", "path", path, "records count", count)
	} else {
		d.logger.Info("Database connected successfully", "path", path, "records count", count)
	}
	return nil
}

func (d *CustomDataset) Init(ctx context.Context, logger logr.Logger, path string, url string, useInMemory bool) error {
	d.logger = logger
	if path == "" {
		return errors.New("no dataset path provided")
	}
	d.hasWarned = false
	if url == "" {
		d.logger.Info("Using dataset from", "path", path)
		return d.connectToDB(path, useInMemory)
	}
	_, err := os.Stat(path)
	if err != nil {
		// file does not exist, download it
		err = d.downloadDataset(ctx, url, path)
		if err != nil {
			// if the file is created but incomplete, remove it
			if _, statErr := os.Stat(path); statErr == nil {
				cerr := os.Remove(path)
				if cerr != nil {
					d.logger.Error(cerr, "failed to remove incomplete file after download")
				}
			}
			return fmt.Errorf("failed to download dataset: %w", err)
		}
	}
	d.logger.Info("Using dataset path", "dataset-path", path)

	return d.connectToDB(path, useInMemory)
}

func (d *CustomDataset) Close() error {
	// Release db lock (only for file-based databases)
	_, err := d.db.Exec("ROLLBACK;")
	if err != nil {
		if cerr := d.db.Close(); cerr != nil {
			d.logger.Error(cerr, "failed to close database after failing to acquire exclusive lock")
		}
		d.db = nil
		return fmt.Errorf("failed to release exclusive lock: %w", err)
	}

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
	tokens, err := d.GenerateTokens(req, nTokensToGen, finishReason)
	return tokens, finishReason, err
}

func (d *CustomDataset) query(query string, nTokens int) ([][]string, error) {
	rows, err := d.db.Query(query)
	if err != nil {
		if !d.hasWarned {
			d.logger.Error(err, "Failed to query database. Ensure dataset file is still valid. Will generate random tokens instead.")
			d.hasWarned = true
		}
		return [][]string{GenPresetRandomTokens(nTokens)}, nil
	}
	defer func() {
		if cerr := rows.Close(); cerr != nil {
			d.logger.Error(cerr, "failed to close rows after query")
		}
	}()
	return unmarshalAllRecords(rows)
}

func (d *CustomDataset) GenerateTokens(req openaiserverapi.CompletionRequest, nTokens int, finishReason string) ([]string, error) {
	// query by prompt hash first
	promptHash := d.GetPromptHash(req)
	promptHashHex := d.GetPromptHashHex(promptHash)
	query := "SELECT " + genTokensCol + " FROM " + tableName + " WHERE " + promptHashCol + "=X'" + promptHashHex + "';"
	tokensList, err := d.query(query, nTokens)

	// filter out results according to finish reason
	var filteredTokensList [][]string
	if finishReason != LengthFinishReason && finishReason != StopFinishReason {
		d.logger.Error(errors.New("unknown finish reason"), "Unexpected finish reason", "reason", finishReason)
	}
	for _, tokens := range tokensList {
		if finishReason == StopFinishReason && len(tokens) <= nTokens {
			filteredTokensList = append(filteredTokensList, tokens)
		} else if finishReason == LengthFinishReason && len(tokens) == nTokens {
			filteredTokensList = append(filteredTokensList, tokens)
		}
	}
	tokensList = filteredTokensList

	if err != nil || len(filteredTokensList) == 0 {
		switch finishReason {
		case LengthFinishReason:
			query = "SELECT " + genTokensCol + " FROM " + tableName + " WHERE " + nGenTokensCol + "=" + strconv.Itoa(nTokens) + ";"
			tokensList, err = d.query(query, nTokens)
		case StopFinishReason:
			query = "SELECT " + genTokensCol + " FROM " + tableName + " WHERE " + nGenTokensCol + "<=" + strconv.Itoa(nTokens) + ";"
			tokensList, err = d.query(query, nTokens)
		}
	}

	if err != nil || len(tokensList) == 0 {
		// if both queries fail or return no results, generate random tokens
		return GenPresetRandomTokens(nTokens), nil
	}
	if d.hasWarned {
		d.hasWarned = false
	}
	randIndex := common.RandomInt(0, len(tokensList)-1)
	return tokensList[randIndex], nil
}
