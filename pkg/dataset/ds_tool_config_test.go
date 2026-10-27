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
	"slices"
	"testing"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
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

func TestDSToolStoreToSQLitePromptHashIndex(t *testing.T) {
	tests := []struct {
		name      string
		tableName string
		// storedName is the table name SQLite stores, without quotes or schema
		storedName     string
		indexedColumns []string
	}{
		{name: "default table name", tableName: "llmd", storedName: "llmd", indexedColumns: []string{promptHashCol}},
		{name: "quoted table name", tableName: `"my-data"`, storedName: "my-data", indexedColumns: []string{promptHashCol}},
		// CREATE INDEX cannot name a schema-qualified table, the records are stored without the index
		{name: "schema-qualified table name", tableName: "main.llmd", storedName: "llmd"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := NewDefaultDSToolConfiguration()
			cfg.outputPath = t.TempDir()
			cfg.tableName = tt.tableName
			tool := &DatasetTool{config: cfg, sqlHelper: newSqliteHelper(cfg.tableName, logr.Discard()), logger: logr.Discard()}
			records := []outputRecord{{PromptHash: []byte{1}, NumGenTokens: 1, GenTokens: api.Tokenized{Strings: []string{"a"}, Tokens: []uint32{1}}}}
			if err := tool.storeToSQLite(context.Background(), records); err != nil {
				t.Fatal(err)
			}
			db, err := sql.Open("sqlite3", "file:"+cfg.getOutputDBFullFileName()+"?mode=ro")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := db.Close(); err != nil {
					t.Error(err)
				}
			})
			var count int
			if err := db.QueryRow("SELECT COUNT(*) FROM " + tt.tableName).Scan(&count); err != nil {
				t.Fatal(err)
			}
			if count != len(records) {
				t.Fatalf("records count = %d; want %d", count, len(records))
			}
			rows, err := db.Query("SELECT i.name FROM sqlite_master AS m JOIN pragma_index_info(m.name) AS i WHERE m.type = 'index' AND m.tbl_name = ?", tt.storedName)
			if err != nil {
				t.Fatal(err)
			}
			defer rows.Close() //nolint:errcheck
			var columns []string
			for rows.Next() {
				var column string
				if err := rows.Scan(&column); err != nil {
					t.Fatal(err)
				}
				columns = append(columns, column)
			}
			if err := rows.Err(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(columns, tt.indexedColumns) {
				t.Fatalf("indexed columns = %v; want %v", columns, tt.indexedColumns)
			}
		})
	}
}
