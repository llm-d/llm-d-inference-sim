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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-logr/logr"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

type conversation struct {
	Role  string `json:"from"`
	Value string `json:"value"`
}

type datasetRecord struct {
	ID            string         `json:"id"`
	Conversations []conversation `json:"conversations"`
}

type genTokens struct {
	TokenStrings []string `json:"strings"`
	TokenNumbers []uint32 `json:"numbers"`
}

type outputRecord struct {
	PromptHash   []byte    `json:"prompt_hash"`
	NumGenTokens int       `json:"n_gen_tokens"`
	GenTokens    genTokens `json:"gen_tokens"`
	InputText    string    `json:"input_text"` // input text for reference and debugging
	Generated    string    `json:"generated"`  // generated text for reference and debugging
}

type DatasetTool struct {
	config    *DSToolConfiguration
	tokenizer tokenizer.Tokenizer
	sqlHelper *sqliteHelper
	logger    logr.Logger
}

func NewDatasetTool(config *DSToolConfiguration, logger logr.Logger) (*DatasetTool, error) {
	t, err := tokenizer.NewHFTokenizer(config.model, config.tokenizersCacheDir)
	if err != nil {
		return nil, err
	}
	return &DatasetTool{
		config:    config,
		tokenizer: t,
		sqlHelper: newSqliteHelper(logger),
		logger:    logger,
	}, nil
}

func (dt *DatasetTool) Run(ctx context.Context) error {
	records, err := dt.loadData(ctx)
	if err != nil {
		dt.logger.Error(err, "failed to load data")
		return err
	}
	dt.logger.Info("Loaded records", "count", len(records))

	// convert loaded original dataset records to db records
	outputRecs, err := dt.convertToOutputRecords(records)
	if err != nil {
		dt.logger.Error(err, "failed to convert dataset records to database records")
		return err
	}
	dt.logger.Info("Loaded db records", "count", len(outputRecs))

	err = dt.storeRecordsToSQLite(ctx, outputRecs)
	if err != nil {
		dt.logger.Error(err, "failed to store dataset to sqlite db")
		return err
	}
	err = dt.storeRecordsToJson(outputRecs)
	if err != nil {
		dt.logger.Error(err, "failed to store dataset to json debug file")
		return err
	}

	return nil
}

func (dt *DatasetTool) loadData(ctx context.Context) ([]datasetRecord, error) {
	var data []byte
	var err error
	fullPath := ""

	if dt.config.hfRepo != "" {
		fullPath = dt.config.hfRepo + "/" + dt.config.file
		// HuggingFace mode
		dt.logger.Info("Loading HF dataset", "hf file", fullPath)
		client := newHFClient(dt.config.token)
		data, err = client.downloadFile(ctx, dt.config.hfRepo, dt.config.file)
	} else {
		fullPath = filepath.Join(dt.config.localPath, dt.config.file)
		// Local file mode
		dt.logger.Info("Loading local files from a folder", "local file", fullPath)
		data, err = loadLocalFile(fullPath)
	}

	if err != nil {
		dt.logger.Error(err, "failed to load", "file", fullPath)
		return nil, err
	}

	records, err := parseJson(data)
	if err != nil {
		dt.logger.Error(err, "failed to parse", "file", fullPath)
		return nil, err
	}

	dt.logger.Info("Loaded records", "count", len(records), "file", fullPath)
	if len(records) >= dt.config.maxRecords {
		records = records[:dt.config.maxRecords]
	}

	return records, nil
}

func (dt *DatasetTool) convertToOutputRecords(dsRecords []datasetRecord) ([]outputRecord, error) {
	resultRecs := []outputRecord{}

	for _, dsRecord := range dsRecords {
		conversationIndex := 0
		chatRequest := openaiserverapi.ChatCompletionRequest{}
		chatRequest.Messages = []openaiserverapi.Message{}
		prevOutput := ""

		// read conversations in pairs
		for conversationIndex < len(dsRecord.Conversations)-1 {
			if !dt.validateConversationRole(dsRecord, conversationIndex) {
				break
			}

			records := dt.conversationToRecords(dsRecord.Conversations[conversationIndex], dsRecord.Conversations[conversationIndex+1],
				prevOutput, &chatRequest)
			resultRecs = append(resultRecs, records...)
			// save the output for the next iteration
			prevOutput = dsRecord.Conversations[conversationIndex+1].Value
			conversationIndex += 2
		}
	}

	return resultRecs, nil
}

func (dt *DatasetTool) conversationToRecords(conversation1, conversation2 conversation,
	prevOutput string, chatRequest *openaiserverapi.ChatCompletionRequest) []outputRecord {
	result := []outputRecord{}
	input := conversation1.Value

	textRequest := openaiserverapi.TextCompletionRequest{
		Prompt: input,
	}

	// add previous assistant message
	if len(prevOutput) > 0 {
		chatRequest.Messages = append(chatRequest.Messages,
			openaiserverapi.Message{Role: openaiserverapi.RoleAssistant,
				Content: openaiserverapi.Content{Raw: prevOutput}})
	}
	// add current user message
	chatRequest.Messages = append(chatRequest.Messages, openaiserverapi.Message{
		Role:    openaiserverapi.RoleUser,
		Content: openaiserverapi.Content{Raw: input},
	})

	// create db record for /completions (without the messages concatunation)
	inputText := textRequest.GetPrompt()
	if rec, err := dt.createJsonRecord(inputText); err != nil {
		return []outputRecord{}
	} else {
		result = append(result, *rec)
	}

	// create db record for /chat/completions with all messages till now
	inputText = chatRequest.GetPrompt()
	if rec, err := dt.createJsonRecord(inputText); err != nil {
		return []outputRecord{}
	} else {
		result = append(result, *rec)
	}

	return result
}

func (dt *DatasetTool) createJsonRecord(inputText string) (*outputRecord, error) {
	tokens, textTokens, err := dt.tokenizer.Encode(inputText, dt.config.model)
	if err != nil {
		dt.logger.Error(err, "failed to encode conversation, skip it")
		return nil, err
	}
	rec := outputRecord{
		// TODO hash the input
		PromptHash:   []byte{}, //hash(dsRecord.Conversations[conversationIndex].),
		NumGenTokens: len(tokens),
		GenTokens:    genTokens{TokenStrings: textTokens, TokenNumbers: tokens},
		InputText:    inputText,
		Generated:    strings.Join(textTokens, ""),
	}

	return &rec, nil
}

func (dt *DatasetTool) validateConversationRole(dsRecord datasetRecord, conversationIndex int) bool {
	if dsRecord.Conversations[conversationIndex].Role != "human" {
		dt.logger.Error(nil, "Invalid role in ds record", "index", conversationIndex,
			"expected role", "human",
			"real", dsRecord.Conversations[conversationIndex].Role)
		return false
	}
	if dsRecord.Conversations[conversationIndex+1].Role != "gpt" {
		dt.logger.Error(nil, "Invalid role in ds record", "index", conversationIndex+1,
			"expected role", "gpt",
			"real", dsRecord.Conversations[conversationIndex+1].Role)
		return false
	}
	return true
}

func parseJson(data []byte) ([]datasetRecord, error) {
	var records []datasetRecord

	if err := json.Unmarshal([]byte(data), &records); err != nil {
		return nil, fmt.Errorf("unmarshal: %v", err)
	}

	return records, nil
}

// loadLocalFile loads file
func loadLocalFile(fullPath string) ([]byte, error) {
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return nil, errors.Join(err, fmt.Errorf("cannot read file %s", fullPath))
	}
	return data, nil
}

// creates the database table and stores the records
func (dt *DatasetTool) storeRecordsToSQLite(ctx context.Context, records []outputRecord) error {
	dbPath := dt.config.getOutputDBFullFileName()
	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		return errors.Join(err, fmt.Errorf("cannot open database %s", dbPath))
	}
	defer db.Close()

	// Verify connection with context
	if err := db.PingContext(ctx); err != nil {
		return fmt.Errorf("failed to ping database: %w", err)
	}

	// Create table if not exists
	if _, err := db.ExecContext(ctx, dt.sqlHelper.getCreateTableQuery()); err != nil {
		return fmt.Errorf("failed to create table: %w", err)
	}
	dt.logger.Info("Table created successfully", "table", tableName)

	// Insert records
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.PrepareContext(ctx, dt.sqlHelper.getInsertQuery())
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for _, record := range records {
		// Check for cancellation
		select {
		case <-ctx.Done():
			return fmt.Errorf("operation cancelled: %w", ctx.Err())
		default:
		}

		// Marshal genTokens slice to JSON
		genTokensJSON, err := json.Marshal(record.GenTokens)
		if err != nil {
			return fmt.Errorf("failed to marshal gen_tokens: %w", err)
		}

		if _, err := stmt.ExecContext(ctx, record.PromptHash, genTokensJSON, record.NumGenTokens); err != nil {
			return fmt.Errorf("failed to insert record: %w", err)
		}
	}
	dt.logger.Info("Records stored sucessfully", "count", len(records))
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

func (dt *DatasetTool) storeRecordsToJson(records []outputRecord) error {
	filePath := dt.config.getOutputJsonFullFileName()
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	dt.logger.Info("Storing records to JSON", "file", filePath)
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ") // Pretty print

	if err := encoder.Encode(records); err != nil {
		return fmt.Errorf("failed to encode records to JSON: %w", err)
	}

	return nil
}

// TODO - read table name fro configuration
