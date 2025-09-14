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

package llmdinferencesim

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/go-logr/logr"
	_ "github.com/mattn/go-sqlite3"
)

type Dataset struct {
	db     *sql.DB
	logger logr.Logger
}

func (d *Dataset) downloadDataset(url string, savePath string) error {
	// Set up signal handling for Ctrl+C (SIGINT)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigs)

	// Goroutine to listen for signal
	go func() {
		<-sigs
		d.logger.Info("Interrupt signal received, cancelling download...")
		cancel()
	}()

	out, err := os.Create(savePath)
	if err != nil {
		return err
	}
	defer func() {
		cerr := out.Close()
		if cerr != nil {
			d.logger.Error(cerr, "failed to close file after download")
		}
	}()

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
		hasShownSpeed: false,
	}

	written, err := io.Copy(out, pr)
	if err != nil {
		// Remove incomplete file
		cerr := os.Remove(savePath)
		if cerr != nil {
			d.logger.Error(cerr, "failed to remove incomplete file after download")
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
			d.logger.Error(cerr, "failed to remove empty file after download")
		}
		return errors.New("downloaded file is empty")
	}

	// Ensure file is fully flushed and closed before returning success
	if err := out.Sync(); err != nil {
		cerr := os.Remove(savePath)
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
	total      int64
	downloaded int64
	startTime  time.Time
	lastPct    int
	logger     logr.Logger
	ctx        context.Context
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

func (d *Dataset) connectToDB(path string) error {
	// check if file exists
	_, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("database file does not exist: %w", err)
	}
	d.db, err = sql.Open("sqlite3", path)
	if err != nil {
		return fmt.Errorf("failed to open database: %w", err)
	}
	// Test the connection

	return nil
}

func (d *Dataset) Init(path string, url string, savePath string) error {
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
			d.logger.Info("Downloading dataset from URL", "url", url, "to", savePath)
			err = d.downloadDataset(url, savePath)
			if err != nil {
				return fmt.Errorf("failed to download dataset: %w", err)
			}
		}
		d.logger.Info("Using dataset from", "path", savePath)

		return d.connectToDB(savePath)
	}
	return errors.New("no dataset path or url provided")
}

func (d *Dataset) Close() error {
	if d.db != nil {
		return d.db.Close()
	}
	return nil
}
