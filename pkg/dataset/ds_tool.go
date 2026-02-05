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

	"github.com/go-logr/logr"
	openaiserverapi "github.com/llm-d/llm-d-inference-sim/pkg/openai-server-api"
	"github.com/llm-d/llm-d-inference-sim/pkg/tokenizer"
)

// conversation in the source dataset record
type conversation struct {
	Role  string `json:"from"`
	Value string `json:"value"`
}

// the source dataset record
type datasetRecord struct {
	ID            string         `json:"id"`
	Conversations []conversation `json:"conversations"`
}

// record of the output dataset
type outputRecord struct {
	PromptHash   []byte                    `json:"prompt_hash"`
	NumGenTokens int                       `json:"n_gen_tokens"`
	GenTokens    openaiserverapi.Tokenized `json:"gen_tokens"`
	InputText    string                    `json:"input_text"` // input text for reference and debugging
	Generated    string                    `json:"generated"`  // generated text for reference and debugging
}

// DatasetTool the dataset tool
type DatasetTool struct {
	config    *DSToolConfiguration
	tokenizer tokenizer.Tokenizer
	sqlHelper *sqliteHelper
	logger    logr.Logger
}

// NewDatasetTool creates DatasetTool instance based on the given parameters
func NewDatasetTool(config *DSToolConfiguration, logger logr.Logger) (*DatasetTool, error) {
	t, err := tokenizer.NewHFTokenizer(config.model, config.tokenizersCacheDir)
	if err != nil {
		return nil, err
	}
	return &DatasetTool{
		config:    config,
		tokenizer: t,
		sqlHelper: newSqliteHelper(config.tableName, logger),
		logger:    logger,
	}, nil
}

// Run runs the dataset creation tool
// It reads the input huggingface dataset, which could be downloaded from the HF site
// or stored locally. This kind of dataset contains conversations between human and gpt.
// Output dataset contains generated responses for prompts in the source dataset.
// Responses are created for both formats: completions and chat completions.
// Example:
// Source records:
// [{"id": "test", "conversations": [
//
//	{"from": "human", "value": "human q1"},
//	{"from": "gpt", "value": "gpt a1"},
//	{"from": "human", "value": "human q2"},
//	{"from": "gpt", "value": "gpt a2" }]}]
//
// Output records:
// [{"prompt_hash": "OZ5Edy+9rw0CsSMabW2TwSxR78jJGYRVRWtz8SXRm6U=",
//
//	 "n_gen_tokens": 4,
//	 "gen_tokens": {"strings": ["g","pt"," a","1"], "numbers": [...]},
//	 "input_text": "human q1",
//	 "generated": "gpt a1"},
//	{"prompt_hash": "8eh+o90xEiD3eJ7fYJxpr3i1J6H8qx1GYnKpen8Jktg=",
//	 "n_gen_tokens": 4,
//	 "gen_tokens": {"strings": ["g","pt"," a","1"], "numbers": [...]},
//	 "input_text": "### user:\nhuman q1\n",
//	 "generated": "gpt a1"},
//	{"prompt_hash": "IA09arbBXHzUc87MBMHVHyrOL7tOHaAjQurbzggNJoY=",
//	 "n_gen_tokens": 4,
//	 "gen_tokens": {"strings": ["gp","t"," a","2"], "numbers": [...]},
//	 "input_text": "human q2",
//	 "generated": "gpt a2"},
//	{"prompt_hash": "J+hRSEBht2WjJ2/4Mq+HfNCWf2VvxHaP11LnwJ7yHWE=",
//	 "n_gen_tokens": 4,
//	 "gen_tokens": {"strings": ["gp","t"," a","2"], "numbers": [...]},
//	 "input_text": "### user:\nhuman q1\n### assistant:\ngpt a1\n### user:\nhuman q2\n",
//	 "generated": "gpt a2"}]
func (dt *DatasetTool) Run(ctx context.Context) error {
	sourceRecs, err := dt.loadData(ctx)
	if err != nil {
		dt.logger.Error(err, "failed to load the source dataset")
		return err
	}
	dt.logger.Info("Loaded source dataset records", "count", len(sourceRecs))

	// convert loaded original dataset records to output records
	outputRecs := dt.toOutputRecords(sourceRecs)
	dt.logger.Info("Created output records", "count", len(outputRecs))

	err = dt.storeToSQLite(ctx, outputRecs)
	if err != nil {
		dt.logger.Error(err, "failed to store dataset to sqlite db")
		return err
	}
	err = dt.storeToJson(outputRecs)
	if err != nil {
		dt.logger.Error(err, "failed to store dataset to json debug file")
		return err
	}

	if err = generateCardFile(dt.config.model, dt.config.tableName, dt.config.hfRepo, dt.config.inputFile,
		dt.config.getOutputCardFullFileName(), len(sourceRecs), len(outputRecs)); err != nil {
		dt.logger.Error(err, "failed to store dataset card file")
		return err
	}

	return nil
}

// loads source dataset data, both local or from HF
func (dt *DatasetTool) loadData(ctx context.Context) ([]datasetRecord, error) {
	var sourceData []byte
	var err error
	fullPath := ""

	if dt.config.hfRepo != "" {
		// HuggingFace mode
		fullPath = dt.config.hfRepo + "/" + dt.config.inputFile
		dt.logger.Info("Loading HF dataset", "path", fullPath)
		client := newHFClient(dt.config.token)
		sourceData, err = client.downloadFile(ctx, dt.config.hfRepo, dt.config.inputFile)
	} else {
		// Local file mode
		fullPath = filepath.Join(dt.config.localPath, dt.config.inputFile)
		dt.logger.Info("Loading local files from a folder", "local file", fullPath)
		sourceData, err = loadLocalFile(fullPath)
	}

	if err != nil {
		dt.logger.Error(err, "failed to load source dataset", "path", fullPath)
		return nil, err
	}

	records, err := parseSourceJson(sourceData)
	if err != nil {
		dt.logger.Error(err, "failed to parse", "file", fullPath)
		return nil, err
	}

	dt.logger.Info("Loaded source records", "count", len(records), "path", fullPath)
	if len(records) >= dt.config.maxRecords {
		records = records[:dt.config.maxRecords]
	}

	return records, nil
}

// toOutputRecords converts source dataset records to output records
func (dt *DatasetTool) toOutputRecords(dsRecords []datasetRecord) []outputRecord {
	resultRecs := []outputRecord{}

	for index, dsRecord := range dsRecords {
		chatRequest := openaiserverapi.ChatCompletionRequest{}
		chatRequest.Messages = []openaiserverapi.Message{}

		// read conversations in pairs
		for conversationIndex := 0; conversationIndex < len(dsRecord.Conversations)-1; conversationIndex += 2 {
			if !dt.validConversationRole(dsRecord, conversationIndex) {
				break
			}

			records, err := dt.conversationToOutputRecords(dsRecord.Conversations[conversationIndex].Value,
				dsRecord.Conversations[conversationIndex+1].Value, &chatRequest)
			resultRecs = append(resultRecs, records...)

			if err != nil {
				dt.logger.Error(err, "failed to encode conversation output, skip it", "index", index)
				continue
			}
		}
	}

	return resultRecs
}

// conversationToOutputRecords creates output records from the given parameters
// updates the given chatRequest with a new step in the conversation
func (dt *DatasetTool) conversationToOutputRecords(userTxt, assistantTxt string,
	chatRequest *openaiserverapi.ChatCompletionRequest) ([]outputRecord, error) {
	result := []outputRecord{}

	// create completions request
	textRequest := openaiserverapi.TextCompletionRequest{
		Prompt: userTxt,
	}

	// add current user message
	chatRequest.Messages = append(chatRequest.Messages, openaiserverapi.Message{
		Role:    openaiserverapi.RoleUser,
		Content: openaiserverapi.Content{Raw: userTxt},
	})

	// create db record for /completions (without the messages concatenation)
	inputText := textRequest.GetPrompt()
	if rec, err := dt.createOutputRecord(inputText, assistantTxt); err != nil {
		return nil, err
	} else {
		result = append(result, *rec)
	}

	// create db record for /chat/completions with all messages till now
	// TODO - templatize!
	inputText = chatRequest.GetFullPrompt()
	if rec, err := dt.createOutputRecord(inputText, assistantTxt); err != nil {
		return nil, err
	} else {
		result = append(result, *rec)
	}

	// add answer for this turn to be ready for the next question
	chatRequest.Messages = append(chatRequest.Messages,
		openaiserverapi.Message{Role: openaiserverapi.RoleAssistant,
			Content: openaiserverapi.Content{Raw: assistantTxt}})

	return result, nil
}

// createOutputRecord creates an output record based on the given parameters
func (dt *DatasetTool) createOutputRecord(inputText, generatedText string) (*outputRecord, error) {
	inputTokens, _, err := dt.tokenizer.Encode(inputText, dt.config.model)
	if err != nil {
		return nil, errors.Join(err, fmt.Errorf("input tokenization failed (%s)", inputText))
	}
	generatedTokens, genTextTokens, err := dt.tokenizer.Encode(generatedText, dt.config.model)
	if err != nil {
		return nil, errors.Join(err, fmt.Errorf("output tokenization failed (%s)", generatedText))
	}

	rec := outputRecord{
		PromptHash:   getInputHash(inputTokens),
		NumGenTokens: len(generatedTokens),
		GenTokens:    openaiserverapi.Tokenized{Strings: genTextTokens, Tokens: generatedTokens},
		InputText:    inputText,
		Generated:    generatedText,
	}

	return &rec, nil
}

// checks validity of the role of the pair of records defined by conversationIndex
// first record should be human and the second is gpt
func (dt *DatasetTool) validConversationRole(dsRecord datasetRecord, conversationIndex int) bool {
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

// creates the database table and stores the given records
func (dt *DatasetTool) storeToSQLite(ctx context.Context, records []outputRecord) error {
	dbPath := dt.config.getOutputDBFullFileName()
	dt.logger.Info("Going to store records to DB", "path", dbPath)
	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		return errors.Join(err, fmt.Errorf("cannot open database %s", dbPath))
	}
	defer func() {
		_ = db.Close()
	}()

	// Verify connection with context
	if err := db.PingContext(ctx); err != nil {
		return fmt.Errorf("failed to ping database: %w", err)
	}

	// Create table if not exists
	if _, err := db.ExecContext(ctx, dt.sqlHelper.getCreateTableQuery()); err != nil {
		return fmt.Errorf("failed to create table: %w", err)
	}
	dt.logger.Info("Table created successfully", "table", dt.config.tableName)

	// Insert records
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	stmt, err := tx.PrepareContext(ctx, dt.sqlHelper.getInsertQuery())
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer func() {
		_ = stmt.Close()
	}()

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
	dt.logger.Info("Records stored successfully", "count", len(records))
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

func (dt *DatasetTool) storeToJson(records []outputRecord) error {
	filePath := dt.config.getOutputJsonFullFileName()
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer func() {
		_ = file.Close()
	}()

	dt.logger.Info("Storing records to JSON", "file", filePath)
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ") // Pretty print

	if err := encoder.Encode(records); err != nil {
		return fmt.Errorf("failed to encode records to JSON: %w", err)
	}

	return nil
}
