/*
Copyright 2026 The llm-d-inference-sim Authors.

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
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-logr/logr"
)

func TestDSToolConfigurationOutput(t *testing.T) {
	tests := []struct {
		name       string
		args       []string
		outputPath string
		tableName  string
	}{
		{name: "defaults", tableName: "llmd"},
		{name: "table name after output path", args: []string{"--output-path", "output", "--table-name", "custom"}, outputPath: "output", tableName: "custom"},
		{name: "output path after table name", args: []string{"--table-name", "custom", "--output-path", "output"}, outputPath: "output", tableName: "custom"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Chdir(t.TempDir())
			originalArgs := os.Args
			t.Cleanup(func() { os.Args = originalArgs })
			os.Args = append([]string{"dataset-tool", "--model", "test", "--local-path", ".", "--input-file", "input.json", "--render-url", "http://localhost:8082"}, tt.args...)
			cfg := NewDefaultDSToolConfiguration()
			if err := cfg.LoadConfig(); err != nil {
				t.Fatal(err)
			}
			if cfg.outputPath != tt.outputPath || cfg.tableName != tt.tableName {
				t.Fatalf("output path = %q, table name = %q; want %q, %q", cfg.outputPath, cfg.tableName, tt.outputPath, tt.tableName)
			}
			if tt.outputPath != "" {
				if err := os.Mkdir(tt.outputPath, 0755); err != nil {
					t.Fatal(err)
				}
			}
			tool := &DatasetTool{config: cfg, sqlHelper: newSqliteHelper(cfg.tableName, logr.Discard()), logger: logr.Discard()}
			if err := tool.storeToSQLite(context.Background(), nil); err != nil {
				t.Fatal(err)
			}
			dbPath := filepath.Join(tt.outputPath, "inference-sim-dataset.sqlite3")
			if _, err := os.Stat(dbPath); err != nil {
				t.Fatal(err)
			}
			db, err := sql.Open("sqlite3", "file:"+dbPath+"?mode=ro")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := db.Close(); err != nil {
					t.Error(err)
				}
			})
			var tableName string
			if err := db.QueryRow("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", tt.tableName).Scan(&tableName); err != nil {
				t.Fatal(err)
			}
		})
	}
}
