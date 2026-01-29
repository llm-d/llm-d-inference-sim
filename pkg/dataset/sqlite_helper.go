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
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
)

// use constants for expected column names and types
const (
	tableName         = "llmd"
	idCol             = "id"
	promptHashCol     = "prompt_hash"
	genTokensCol      = "gen_tokens"
	nGenTokensCol     = "n_gen_tokens"
	idColType         = "INTEGER"
	promptHashColType = "BLOB"
	genTokensColType  = "JSON"
	nGenTokensColType = "INTEGER"
)

type sqliteHelper struct {
	logger logr.Logger
	db     *sql.DB
}

func newSqliteHelper(logger logr.Logger) *sqliteHelper {
	return &sqliteHelper{logger: logger}
}

func (s *sqliteHelper) connectToDB(path string, useInMemory bool, needVerification bool) error {
	if s.db != nil {
		err := s.db.Close()
		if err != nil {
			s.logger.Error(err, "failed to close existing database connection")
		}
		s.db = nil
	}
	// check if file exists
	_, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("database file does not exist: %w", err)
	}

	if useInMemory {
		err = s.loadDatabaseInMemory(path)
		if err != nil {
			return err
		}
	} else {
		// Use file-based database (original behavior)
		s.db, err = sql.Open("sqlite3", "file:"+path+"?mode=ro")
		if err != nil {
			return fmt.Errorf("failed to open database: %w", err)
		}

		// Check if there are other connections to the database
		_, err = s.db.Exec("BEGIN EXCLUSIVE;")
		if err != nil {
			if closeErr := s.db.Close(); closeErr != nil {
				s.logger.Error(closeErr, "failed to close database after failing to acquire exclusive lock")
			}
			s.db = nil
			return fmt.Errorf("database is locked or has other active connections: %w", err)
		}
	}

	count := 0
	if needVerification {
		err = s.verifyDB()
		if err != nil {
			return fmt.Errorf("failed to verify database: %w", err)
		}

		count, err = s.getRecordsCount()
		if err != nil {
			s.logger.Error(err, "failed to get records count")
			return fmt.Errorf("failed to query database: %w", err)
		}
	}

	if useInMemory {
		s.logger.V(logging.INFO).Info("In-memory database connected successfully", "path", path, "records count", count)
	} else {
		s.logger.V(logging.INFO).Info("Database connected successfully", "path", path, "records count", count)
	}
	return nil
}

func (s *sqliteHelper) loadDatabaseInMemory(path string) error {
	s.logger.V(logging.INFO).Info("Loading database into memory...")
	start := time.Now()

	// Create in-memory database
	var err error
	s.db, err = sql.Open("sqlite3", ":memory:")
	if err != nil {
		return fmt.Errorf("failed to create in-memory database: %w", err)
	}

	// Use ATTACH to copy the database
	attachSQL := fmt.Sprintf("ATTACH DATABASE '%s' AS source", path)
	_, err = s.db.Exec(attachSQL)
	if err != nil {
		if closeErr := s.db.Close(); closeErr != nil {
			s.logger.Error(closeErr, "failed to close in-memory database after attach failure")
		}
		s.db = nil
		return fmt.Errorf("failed to attach source database: %w", err)
	}

	// Copy the table structure first
	_, err = s.db.Exec(s.getCreateTableQuery())
	if err != nil {
		if closeErr := s.db.Close(); closeErr != nil {
			s.logger.Error(closeErr, "failed to close in-memory database after create table failure")
		}
		s.db = nil
		return fmt.Errorf("failed to create table: %w", err)
	}

	// Copy the data
	_, err = s.db.Exec("INSERT INTO " + tableName + " SELECT * FROM source." + tableName)
	if err != nil {
		if closeErr := s.db.Close(); closeErr != nil {
			s.logger.Error(closeErr, "failed to close in-memory database after copy failure")
		}
		s.db = nil
		return fmt.Errorf("failed to copy data: %w", err)
	}

	// Detach the source database
	_, err = s.db.Exec("DETACH DATABASE source")
	if err != nil {
		s.logger.Error(err, "failed to detach source database")
	}

	loadTime := time.Since(start)
	s.logger.V(logging.INFO).Info("Database loaded into memory", "load_time", loadTime.String())
	return nil
}

func (s *sqliteHelper) verifyDB() error {
	rows, err := s.db.Query("PRAGMA table_info(" + tableName + ");")
	if err != nil {
		return fmt.Errorf("failed to query table info for `%s`: %w", tableName, err)
	}
	defer func() {
		if cerr := rows.Close(); cerr != nil {
			s.logger.Error(cerr, "failed to close rows after querying table info")
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

func (s *sqliteHelper) getRecordsCount() (int, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(" + promptHashCol + ") FROM " + tableName + ";").Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to query database: %w", err)
	}
	return count, nil
}

// query runs a SQL query which retrieves response tokens as an array of strings
// returns multuple responses
func (s *sqliteHelper) query(query string) ([][]string, error) {
	rows, err := s.db.Query(query)
	if err != nil {
		s.logger.Error(err, "failed to query database. Ensure dataset file is still valid. Will generate random tokens instead.")
		return nil, err
	}
	defer func() {
		if cerr := rows.Close(); cerr != nil {
			s.logger.Error(cerr, "failed to close rows after query")
			if err == nil {
				err = cerr
			} else {
				err = errors.Join(err, cerr)
			}
		}
	}()
	return unmarshalAllRecords(rows)
}

func unmarshalAllRecords(rows *sql.Rows) ([][]string, error) {
	var responses [][]string

	for rows.Next() {
		var responseJSON string
		if err := rows.Scan(&responseJSON); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		var tokens []string
		if err := json.Unmarshal([]byte(responseJSON), &tokens); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tokens JSON: %w", err)
		}
		responses = append(responses, tokens)
	}
	return responses, nil
}

func (s *sqliteHelper) buildQuery(where string, isRand bool, isLimitOne bool) string {
	query := "SELECT " + genTokensCol + " FROM " + tableName

	if where != "" {
		query += " WHERE " + where
	}

	if isRand {
		query += " ORDER BY RANDOM()"
	}

	if isLimitOne {
		query += " LIMIT 1"
	}
	query += ";"

	return query
}

func (s *sqliteHelper) getResponsesForPrompt(promptHashHex string) ([][]string, error) {
	query := s.buildQuery(promptHashCol+"=X'"+promptHashHex+"'", false, false)
	return s.query(query)
}

func (s *sqliteHelper) getResponsesForLen(maxLen int, isExact bool) ([][]string, error) {
	sign := "<="
	if isExact {
		sign = "="
	}
	query := s.buildQuery(nGenTokensCol+sign+strconv.Itoa(maxLen), true, true)
	return s.query(query)
}

func (s *sqliteHelper) getRandomResponse() ([][]string, error) {
	query := s.buildQuery("", true, true)
	return s.query(query)
}

func (s *sqliteHelper) getCreateTableQuery() string {
	return fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		prompt_hash BLOB NOT NULL,
		gen_tokens JSON NOT NULL,
		n_gen_tokens INTEGER NOT NULL
	)`, tableName)
}

func (s *sqliteHelper) getInsertQuery() string {
	return fmt.Sprintf(`INSERT INTO  %s (prompt_hash, gen_tokens, n_gen_tokens) 
        VALUES (?, ?, ?)`, tableName)
}
