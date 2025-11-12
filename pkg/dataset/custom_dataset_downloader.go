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
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
)

type CustomDatasetDownloader struct {
	logger logr.Logger
}

const (
	progressLogTimeInterval    = 5 * time.Second
	progressLogPercentInterval = 10
)

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

func NewDsDownloader(logger logr.Logger) *CustomDatasetDownloader {
	return &CustomDatasetDownloader{logger: logger}
}

// DownloadDataset downloads dataset from the given url and stores it to the given path
func (d *CustomDatasetDownloader) DownloadDataset(ctx context.Context, url string, path string) error {
	folder := filepath.Dir(path)
	err := os.MkdirAll(folder, 0755)
	if err != nil {
		return fmt.Errorf("failed to create parent directory: %w", err)
	}

	if _, err := os.Stat(path); err == nil {
		// file already exists
		d.logger.V(logging.INFO).Info("Dataset file already exists, should not download: " + path)
		return nil
	}

	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		cerr := out.Close()
		if cerr != nil {
			d.logger.Error(cerr, "failed to close file after download")
			err = errors.Join(err, cerr)
		}
	}()

	d.logger.V(logging.INFO).Info("Using dataset-url", "dataset-url", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer func() {
		cerr := resp.Body.Close()
		if cerr != nil {
			d.logger.Error(cerr, "failed to close response body after download")
			err = errors.Join(err, cerr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("dataset download bad status: %s", resp.Status)
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

	d.logger.V(logging.INFO).Info("Downloaded dataset from %s, stored in %s\n", url, path)
	return nil
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
		pr.logger.V(logging.INFO).Info("Dataset download progress", "%", pct, "speed (MB/s)", speed, "remaining time (s)", remainingTime)
		fmt.Println("Dataset download progress", "%", pct, "speed (MB/s)", speed, "remaining time (s)", remainingTime)
	} else {
		pr.logger.V(logging.INFO).Info("Download completed", "average speed (MB/s)", speed, "total time (s)", elapsedTime)
		fmt.Println("Download completed", "average speed (MB/s)", speed, "total time (s)", elapsedTime)
	}
}
