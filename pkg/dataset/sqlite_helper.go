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
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
	"github.com/ncruces/go-sqlite3/vfs/memdb"
)

// use constants for expected column names and types
const (
	idCol             = "id"
	promptHashCol     = "prompt_hash"
	genTokensCol      = "gen_tokens"
	nGenTokensCol     = "n_gen_tokens"
	idColType         = "INTEGER"
	promptHashColType = "BLOB"
	genTokensColType  = "JSON"
	nGenTokensColType = "INTEGER"
)

// memDBSeq makes the name of each in-memory database unique within the process
var memDBSeq atomic.Uint64

type sqliteHelper struct {
	logger    logr.Logger
	db        *sql.DB
	tableName string
	// memDBName is the memdb database holding the in-memory copy, empty for a file-based database
	memDBName string
}

func newSqliteHelper(tableName string, logger logr.Logger) *sqliteHelper {
	return &sqliteHelper{
		tableName: tableName,
		logger:    logger,
	}
}

func (s *sqliteHelper) connectToDB(path string, useInMemory bool) error {
	s.closeDB()
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

	err = s.verifyDB()
	if err != nil {
		s.closeDB()
		return fmt.Errorf("failed to verify database: %w", err)
	}

	if useInMemory {
		if err := s.ensurePromptHashIndex(); err != nil {
			s.logger.Error(err, "in-memory prompt lookups will scan the whole table")
		}
	} else if indexed, err := s.promptLookupUsesIndex(); err != nil {
		s.logger.Error(err, "failed to check the prompt hash index")
	} else if !indexed {
		s.logger.V(logging.WARN).Info("Dataset has no prompt hash index, prompt lookups will scan the whole table", "path", path)
	}

	count, err := s.getRecordsCount()
	if err != nil {
		s.logger.Error(err, "failed to get records count")
		s.closeDB()
		return fmt.Errorf("failed to query database: %w", err)
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

	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("failed to resolve database path: %w", err)
	}

	// Every pooled connection that opens ":memory:" gets its own empty database,
	// while a memdb database named "/<name>" is shared by all of them. memdb also
	// keeps the data outside SQLite's per-connection memory limit (256 MB).
	// memdb.Create holds a reference, so the data survives the pool closing idle connections.
	s.memDBName = fmt.Sprintf("llmd-dataset-%d", memDBSeq.Add(1))
	memdb.Create(s.memDBName, nil)
	s.db, err = sql.Open("sqlite3", "file:/"+s.memDBName+"?vfs=memdb")
	if err != nil {
		s.closeDB()
		return fmt.Errorf("failed to create in-memory database: %w", err)
	}

	if err := s.copyIntoMemory(absPath); err != nil {
		s.closeDB()
		return err
	}

	loadTime := time.Since(start)
	s.logger.V(logging.INFO).Info("Database loaded into memory", "load_time", loadTime.String())
	return nil
}

// copyIntoMemory copies the dataset table from the file at path into the in-memory database.
// ATTACH applies only to the connection that runs it, so the copy runs on a single connection.
func (s *sqliteHelper) copyIntoMemory(path string) error {
	ctx := context.Background()
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return fmt.Errorf("failed to get in-memory database connection: %w", err)
	}
	defer func() {
		if cerr := conn.Close(); cerr != nil {
			s.logger.Error(cerr, "failed to release in-memory database connection")
		}
	}()

	// vfs=os reads the file from disk; without it the name resolves through memdb to a new, empty database
	source := "file:" + (&url.URL{Path: path}).EscapedPath() + "?vfs=os&mode=ro"
	if _, err := conn.ExecContext(ctx, "ATTACH DATABASE ? AS source", source); err != nil {
		return fmt.Errorf("failed to attach source database: %w", err)
	}

	if _, err := conn.ExecContext(ctx, s.getCreateTableQuery()); err != nil {
		return fmt.Errorf("failed to create table: %w", err)
	}

	if _, err := conn.ExecContext(ctx, "INSERT INTO "+s.tableName+" SELECT * FROM source."+s.tableName); err != nil {
		return fmt.Errorf("failed to copy data: %w", err)
	}

	if err := s.copyIndexes(ctx, conn); err != nil {
		s.logger.Error(err, "failed to copy indexes")
	}

	if _, err := conn.ExecContext(ctx, "DETACH DATABASE source"); err != nil {
		s.logger.Error(err, "failed to detach source database")
	}
	return nil
}

// copyIndexes recreates the indexes of the attached source table in the in-memory database,
// since INSERT ... SELECT copies only the rows
func (s *sqliteHelper) copyIndexes(ctx context.Context, conn *sql.Conn) error {
	rows, err := conn.QueryContext(ctx,
		"SELECT sql FROM source.sqlite_master WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL", s.tableName)
	if err != nil {
		return fmt.Errorf("failed to read source indexes: %w", err)
	}
	var statements []string
	for rows.Next() {
		var statement string
		if err := rows.Scan(&statement); err != nil {
			return errors.Join(fmt.Errorf("failed to read source index: %w", err), rows.Close())
		}
		statements = append(statements, statement)
	}
	if err := errors.Join(rows.Err(), rows.Close()); err != nil {
		return fmt.Errorf("failed to read source indexes: %w", err)
	}

	for _, statement := range statements {
		if _, err := conn.ExecContext(ctx, statement); err != nil {
			return fmt.Errorf("failed to create index (%s): %w", statement, err)
		}
	}
	return nil
}

// ensurePromptHashIndex creates the prompt hash index unless prompt lookups already use an index
func (s *sqliteHelper) ensurePromptHashIndex() error {
	indexed, err := s.promptLookupUsesIndex()
	if err != nil || indexed {
		return err
	}
	if _, err := s.db.Exec(s.getCreatePromptHashIndexQuery()); err != nil {
		return fmt.Errorf("failed to create prompt hash index: %w", err)
	}
	return nil
}

// promptLookupUsesIndex asks the query planner rather than inspecting the indexes,
// since an index on the prompt hash can still be unusable, e.g. a partial one or one with another collation
func (s *sqliteHelper) promptLookupUsesIndex() (bool, error) {
	rows, err := s.db.Query("EXPLAIN QUERY PLAN " + s.getPromptQuery(""))
	if err != nil {
		return false, fmt.Errorf("failed to get the prompt lookup plan: %w", err)
	}
	usesIndex := false
	for rows.Next() {
		var id, parent, notUsed int
		var detail string
		if err := rows.Scan(&id, &parent, &notUsed, &detail); err != nil {
			return false, errors.Join(fmt.Errorf("failed to read the prompt lookup plan: %w", err), rows.Close())
		}
		usesIndex = usesIndex || strings.HasPrefix(detail, "SEARCH")
	}
	if err := errors.Join(rows.Err(), rows.Close()); err != nil {
		return false, fmt.Errorf("failed to read the prompt lookup plan: %w", err)
	}
	return usesIndex, nil
}

// closeDB closes the database and releases its in-memory copy, if any
func (s *sqliteHelper) closeDB() {
	if s.db != nil {
		if err := s.db.Close(); err != nil {
			s.logger.Error(err, "failed to close database")
		}
		s.db = nil
	}
	if s.memDBName != "" {
		memdb.Delete(s.memDBName)
		s.memDBName = ""
	}
}

func (s *sqliteHelper) verifyDB() error {
	rows, err := s.db.Query("PRAGMA table_info(" + s.tableName + ");")
	if err != nil {
		return fmt.Errorf("failed to query table info for `%s`: %w", s.tableName, err)
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
			return fmt.Errorf("missing expected column in %s table: %s", s.tableName, col)
		}
	}

	return nil
}

func (s *sqliteHelper) getRecordsCount() (int, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(" + promptHashCol + ") FROM " + s.tableName + ";").Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to query database: %w", err)
	}
	return count, nil
}

// query runs a SQL query which retrieves response tokens as an array of strings
// returns multiple responses
func (s *sqliteHelper) query(query string) ([]api.Tokenized, error) {
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

func unmarshalAllRecords(rows *sql.Rows) ([]api.Tokenized, error) {
	var responses []api.Tokenized

	for rows.Next() {
		var responseJSON string
		if err := rows.Scan(&responseJSON); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		var tokens api.Tokenized
		if err := json.Unmarshal([]byte(responseJSON), &tokens); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tokens JSON: %w", err)
		}
		responses = append(responses, tokens)
	}
	return responses, nil
}

func (s *sqliteHelper) buildQuery(where string, isRand bool, isLimitOne bool) string {
	query := "SELECT " + genTokensCol + " FROM " + s.tableName

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

func (s *sqliteHelper) getResponsesForPrompt(promptHashHex string) ([]api.Tokenized, error) {
	return s.query(s.getPromptQuery(promptHashHex))
}

func (s *sqliteHelper) getPromptQuery(promptHashHex string) string {
	return s.buildQuery(promptHashCol+"=X'"+promptHashHex+"'", false, false)
}

func (s *sqliteHelper) getResponsesForLen(maxLen int, isExact bool) ([]api.Tokenized, error) {
	sign := "<="
	if isExact {
		sign = "="
	}
	query := s.buildQuery(nGenTokensCol+sign+strconv.Itoa(maxLen), true, true)
	return s.query(query)
}

func (s *sqliteHelper) getRandomResponse() ([]api.Tokenized, error) {
	query := s.buildQuery("", true, true)
	return s.query(query)
}

func (s *sqliteHelper) getCreateTableQuery() string {
	return fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		id INTEGER PRIMARY KEY,
		prompt_hash BLOB NOT NULL,
		gen_tokens JSON NOT NULL,
		n_gen_tokens INTEGER NOT NULL
	)`, s.tableName)
}

// the index name is quoted, since the table name it embeds may be a quoted identifier
func (s *sqliteHelper) getCreatePromptHashIndexQuery() string {
	name := strings.ReplaceAll("idx_"+s.tableName+"_"+promptHashCol, `"`, `""`)
	return fmt.Sprintf(`CREATE INDEX "%s" ON %s (%s)`, name, s.tableName, promptHashCol)
}

func (s *sqliteHelper) getInsertQuery() string {
	return fmt.Sprintf(`INSERT INTO  %s (prompt_hash, gen_tokens, n_gen_tokens) 
        VALUES (?, ?, ?)`, s.tableName)
}
