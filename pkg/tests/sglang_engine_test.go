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

package tests

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

var _ = Describe("sglang engine", func() {
	var client *http.Client

	BeforeEach(func() {
		var err error
		client, err = startServerWithArgs(context.TODO(),
			[]string{"cmd", "--engine", "sglang", "--model", common.TestModelName, "--mode", common.ModeEcho})
		Expect(err).NotTo(HaveOccurred())
	})

	It("Should serve the OpenAI-compatible endpoints", func() {
		message := "This is a test."
		openaiclient, params := getOpenAIClientAndTextParams(client, common.TestModelName, message, false)
		resp, err := openaiclient.Completions.New(context.TODO(), params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Choices).To(HaveLen(1))
		Expect(resp.Choices[0].Text).To(Equal(message))
	})

	It("Should report itself as the owner of the model", func() {
		resp, err := client.Get(baseURL + "/models")
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		Expect(resp.StatusCode).To(Equal(http.StatusOK))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		var models api.ModelsResponse
		Expect(json.Unmarshal(body, &models)).To(Succeed())
		Expect(models.Data).To(HaveLen(1))
		Expect(models.Data[0].OwnedBy).To(Equal("sglang"))
	})

	// No collector is registered for this engine yet, so the route exists and
	// serves nothing. Real sglang mounts /metrics only under --enable-metrics.
	It("Should expose no metrics", func() {
		Expect(sampleMetrics(client)).To(BeEmpty())
	})

	// The Messages API's envelope is the same for every engine; the error type
	// inside it is Anthropic's own vocabulary under this engine, where vLLM keeps
	// its "BadRequestError" spelling.
	It("Should report a Messages API error in Anthropic's vocabulary", func() {
		body := `{"model":"` + common.TestModelName + `","max_tokens":100,"messages":[]}`
		resp, err := client.Post("http://localhost/v1/messages", "application/json",
			bytes.NewBufferString(body))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		respBody, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		var errResp api.MessagesErrorResponse
		Expect(json.Unmarshal(respBody, &errResp)).To(Succeed())
		Expect(errResp.Type).To(Equal(api.MessagesTypeError))
		Expect(errResp.Error.Type).To(Equal("invalid_request_error"))
		Expect(errResp.Error.Message).To(ContainSubstring("messages must not be empty"))
	})

	// The Responses API nests its errors whatever the engine, and this engine
	// spells every error type there the same way whatever the status.
	It("Should report a Responses API error in sglang's format", func() {
		body := `{"model":"no-such-model","input":"hello"}`
		resp, err := client.Post(baseURL+"/responses", "application/json", bytes.NewBufferString(body))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		Expect(resp.StatusCode).To(Equal(http.StatusNotFound))

		respBody, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		var errResp api.ErrorResponse
		Expect(json.Unmarshal(respBody, &errResp)).To(Succeed())
		Expect(errResp.Error.Type).To(Equal("invalid_request_error"))
		Expect(errResp.Error.Code).To(Equal(http.StatusNotFound))
		Expect(errResp.Error.Message).To(ContainSubstring("does not exist"))
	})

	// The Responses API checks a model only when the request sets one.
	It("Should check an empty Responses API model but not an absent one", func() {
		status, body := postJSON(client, "/v1/responses", `{"model":"","input":"hello"}`)
		Expect(status).To(Equal(http.StatusNotFound), string(body))

		status, body = postJSON(client, "/v1/responses", `{"input":"hello"}`)
		Expect(status).To(Equal(http.StatusOK), string(body))
	})

	// sglang puts the error's own fields at the top level, where vLLM wraps them
	// under an "error" key.
	It("Should report an error in sglang's format", func() {
		resp, err := client.Post(baseURL+"/completions", "application/json",
			bytes.NewBufferString(`{"model":"`+common.TestModelName+`","prompt":"hi","max_tokens":0}`))
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		var errBody map[string]any
		Expect(json.Unmarshal(body, &errBody)).To(Succeed())
		Expect(errBody).To(HaveKeyWithValue("object", "error"))
		Expect(errBody).To(HaveKey("message"))
		Expect(errBody).NotTo(HaveKey("error"))
	})

	// Every route vLLM's BindHTTP registers, so that a route added there is
	// either implemented here or reported as absent rather than inherited.
	DescribeTable("should not register a vLLM-only route",
		func(method string, route string) {
			req, err := http.NewRequest(method, "http://localhost"+route, http.NoBody)
			Expect(err).NotTo(HaveOccurred())
			req.Header.Set("Content-Type", "application/json")

			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				Expect(resp.Body.Close()).To(Succeed())
			}()
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound))
		},
		Entry("generate", http.MethodPost, "/inference/v1/generate"),
		Entry("load_lora_adapter", http.MethodPost, "/v1/load_lora_adapter"),
		Entry("unload_lora_adapter", http.MethodPost, "/v1/unload_lora_adapter"),
		Entry("mooncake query", http.MethodGet, "/query"),
		Entry("sleep", http.MethodPost, "/sleep"),
		Entry("wake_up", http.MethodPost, "/wake_up"),
		Entry("is_sleeping", http.MethodGet, "/is_sleeping"),
		Entry("completions render", http.MethodPost, "/v1/completions/render"),
		Entry("chat completions render", http.MethodPost, "/v1/chat/completions/render"),
		Entry("responses render", http.MethodPost, "/v1/responses/render"),
		Entry("messages render", http.MethodPost, "/v1/messages/render"),
		Entry("chat completions derender", http.MethodPost, "/v1/chat/completions/derender"),
		Entry("completions derender", http.MethodPost, "/v1/completions/derender"),
	)

	It("Should serve the routes sglang has: /tokenize and /v1/messages/count_tokens", func() {
		for path, body := range map[string]string{
			"/tokenize": `{"model":"` + common.TestModelName + `","prompt":"hi"}`,
			"/v1/messages/count_tokens": `{"model":"` + common.TestModelName +
				`","messages":[{"role":"user","content":"hi"}]}`,
		} {
			resp, err := client.Post("http://localhost"+path, "application/json", strings.NewReader(body))
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Body.Close()).To(Succeed())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), "path %s", path)
		}
	})
})

// postJSON posts body to route and returns the response's status and body.
func postJSON(client *http.Client, route string, body string) (int, []byte) {
	resp, err := client.Post("http://localhost"+route, "application/json", bytes.NewBufferString(body))
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		Expect(resp.Body.Close()).To(Succeed())
	}()
	respBody, err := io.ReadAll(resp.Body)
	Expect(err).NotTo(HaveOccurred())
	return resp.StatusCode, respBody
}

func expectOpenAIStatus(err error, status int) {
	var openaiError *openai.Error
	Expect(errors.As(err, &openaiError)).To(BeTrue(), "error %v is not an API error", err)
	Expect(openaiError.StatusCode).To(Equal(status))
}

var _ = Describe("sglang engine LoRA", func() {
	var (
		ctx          context.Context
		client       *http.Client
		openaiclient openai.Client
		params       openai.ChatCompletionNewParams
	)

	BeforeEach(func() {
		ctx = context.TODO()
		var err error
		client, err = startServerWithArgs(ctx,
			[]string{"cmd", "--engine", "sglang", "--model", common.TestModelName, "--mode", common.ModeEcho,
				"--max-loras-per-batch", "1", "--time-to-first-token", "500ms",
				"--lora-paths", "lora3=/path/to/lora3", "/path/to/lora4"})
		Expect(err).NotTo(HaveOccurred())
		openaiclient, params = getOpenAIClientAndChatParams(client, common.TestModelName+":lora3", testUserMessage, false)
	})

	It("Should run a request on the adapter its model names, reporting the model as sent", func() {
		resp, err := openaiclient.Chat.Completions.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Model).To(Equal(common.TestModelName + ":lora3"))
		Expect(resp.Choices[0].Message.Content).To(Equal(testUserMessage))
	})

	It("Should run a request on the adapter its lora_path names", func() {
		status, body := postJSON(client, "/v1/completions", `{"model":"`+common.TestModelName+
			`","prompt":"hi","lora_path":"lora3"}`)
		Expect(status).To(Equal(http.StatusOK), string(body))
	})

	It("Should resolve the adapter of each embeddings input", func() {
		status, body := postJSON(client, "/v1/embeddings", `{"model":"`+common.TestModelName+
			`","input":["a","b"],"lora_path":["lora3",null]}`)
		Expect(status).To(Equal(http.StatusOK), string(body))

		status, body = postJSON(client, "/v1/embeddings", `{"model":"`+common.TestModelName+
			`","input":["a","b"],"lora_path":[null,"lora1"]}`)
		Expect(status).To(Equal(http.StatusBadRequest), string(body))
		Expect(string(body)).To(ContainSubstring("Got LoRA adapter that has never been loaded: lora1"))
	})

	It("Should refuse a lora_path of neither form", func() {
		status, body := postJSON(client, "/v1/completions", `{"model":"`+common.TestModelName+
			`","prompt":"hi","lora_path":5}`)
		Expect(status).To(Equal(http.StatusBadRequest))
		var errBody map[string]any
		Expect(json.Unmarshal(body, &errBody)).To(Succeed())
		Expect(errBody).To(HaveKeyWithValue("type", "Bad Request"))
		Expect(errBody).To(HaveKeyWithValue("message", HavePrefix("2 validation errors:")))
	})

	// sglang checks the model only on the Responses API.
	It("Should accept any model, reporting it as sent", func() {
		params.Model = "no-such-model"
		resp, err := openaiclient.Chat.Completions.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
		Expect(resp.Model).To(Equal("no-such-model"))
	})

	It("Should refuse a request naming an adapter never loaded", func() {
		params.Model = common.TestModelName + ":lora1"
		_, err := openaiclient.Chat.Completions.New(ctx, params)
		expectOpenAIStatus(err, http.StatusBadRequest)
	})

	It("Should answer load and unload with sglang's LoRAUpdateOutput", func() {
		status, body := postJSON(client, "/load_lora_adapter", `{"lora_name":"lora1","lora_path":"/path/to/lora1"}`)
		Expect(status).To(Equal(http.StatusOK))
		Expect(body).To(MatchJSON(`{"rid":null,"http_worker_ipc":null,"success":true,"error_message":"",` +
			`"loaded_adapters":{"lora1":"/path/to/lora1","lora3":"/path/to/lora3","/path/to/lora4":"/path/to/lora4"}}`))

		status, body = postJSON(client, "/load_lora_adapter", `{"lora_name":"lora1","lora_path":"/path/to/lora1"}`)
		Expect(status).To(Equal(http.StatusBadRequest))
		Expect(body).To(MatchJSON(`{"rid":null,"http_worker_ipc":null,"success":false,` +
			`"error_message":"Failed to load LoRA adapter lora1 because it is already loaded",` +
			`"loaded_adapters":{"lora1":"/path/to/lora1","lora3":"/path/to/lora3","/path/to/lora4":"/path/to/lora4"}}`))

		status, body = postJSON(client, "/unload_lora_adapter", `{"lora_name":"lora2"}`)
		Expect(status).To(Equal(http.StatusBadRequest))
		Expect(body).To(MatchJSON(`{"rid":null,"http_worker_ipc":null,"success":false,` +
			`"error_message":"LoRA with name lora2 does not exist. ` +
			`Loaded LoRAs: odict_keys(['lora3', '/path/to/lora4', 'lora1'])","loaded_adapters":null}`))

		status, body = postJSON(client, "/unload_lora_adapter", `{}`)
		Expect(status).To(Equal(http.StatusBadRequest))
		missing := `{'type': 'missing', 'loc': ('body', 'lora_name'), 'msg': 'Field required', 'input': {}}`
		Expect(body).To(MatchJSON(`{"object":"error","message":"1 validation error:\n  ` + missing +
			` [` + missing + `]","type":"Bad Request","param":null,"code":400}`))
	})

	It("Should list the adapters with no max_model_len of their own", func() {
		resp, err := client.Get(baseURL + "/models")
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			Expect(resp.Body.Close()).To(Succeed())
		}()
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		var models struct {
			Data []map[string]any `json:"data"`
		}
		Expect(json.Unmarshal(body, &models)).To(Succeed())
		Expect(models.Data).To(HaveLen(3))
		Expect(models.Data[0]).To(HaveKeyWithValue("max_model_len", Not(BeNil())))
		for _, adapter := range models.Data[1:] {
			Expect(adapter).To(HaveKeyWithValue("max_model_len", BeNil()))
			Expect(adapter).To(HaveKeyWithValue("parent", common.TestModelName))
		}
		Expect(models.Data[1]).To(HaveKeyWithValue("id", "lora3"))
		Expect(models.Data[1]).To(HaveKeyWithValue("root", "/path/to/lora3"))
		Expect(models.Data[2]).To(HaveKeyWithValue("id", "/path/to/lora4"))
	})

	It("Should load an unloaded adapter again for a request naming it", func() {
		status, _ := postJSON(client, "/unload_lora_adapter", `{"lora_name":"lora3"}`)
		Expect(status).To(Equal(http.StatusOK))

		_, err := openaiclient.Chat.Completions.New(ctx, params)
		Expect(err).NotTo(HaveOccurred())
	})

	// The unload waits for the request, and the request's batch slot is freed
	// even though its adapter is gone when it completes: with a single slot, a
	// request on another adapter could never run otherwise.
	It("Should unload an adapter once its requests complete, freeing their batch slot", func() {
		start := time.Now()
		done := make(chan error, 1)
		go func() {
			_, err := openaiclient.Chat.Completions.New(ctx, params)
			done <- err
		}()
		time.Sleep(200 * time.Millisecond)

		status, _ := postJSON(client, "/unload_lora_adapter", `{"lora_name":"lora3"}`)
		Expect(status).To(Equal(http.StatusOK))
		Expect(time.Since(start)).To(BeNumerically(">=", 450*time.Millisecond))
		Eventually(done, 5*time.Second).Should(Receive(BeNil()))

		params.Model = common.TestModelName + ":/path/to/lora4"
		_, err := openaiclient.Chat.Completions.New(ctx, params, option.WithRequestTimeout(5*time.Second))
		Expect(err).NotTo(HaveOccurred())
	})

	It("Should check the model on the Responses API, against the adapters too", func() {
		responsesClient, responsesParams := getOpenAIClientAndResponsesParams(client, "lora3", testUserMessage)
		_, err := responsesClient.Responses.New(ctx, responsesParams)
		Expect(err).NotTo(HaveOccurred())

		responsesParams.Model = "lora1"
		_, err = responsesClient.Responses.New(ctx, responsesParams)
		expectOpenAIStatus(err, http.StatusNotFound)
	})
})

var _ = Describe("sglang engine without LoRA", func() {
	var client *http.Client

	BeforeEach(func() {
		var err error
		client, err = startServerWithArgs(context.TODO(),
			[]string{"cmd", "--engine", "sglang", "--model", common.TestModelName, "--mode", common.ModeEcho})
		Expect(err).NotTo(HaveOccurred())
	})

	It("Should refuse a request naming an adapter", func() {
		openaiclient, params := getOpenAIClientAndChatParams(client, common.TestModelName+":lora1",
			testUserMessage, false)
		_, err := openaiclient.Chat.Completions.New(context.TODO(), params)
		expectOpenAIStatus(err, http.StatusBadRequest)
	})

	It("Should refuse to load an adapter", func() {
		status, body := postJSON(client, "/load_lora_adapter", `{"lora_name":"lora1","lora_path":"/path/to/lora1"}`)
		Expect(status).To(Equal(http.StatusBadRequest))
		Expect(body).To(MatchJSON(`{"rid":null,"http_worker_ipc":null,"success":false,` +
			"\"error_message\":\"LoRA is not enabled. Please set `--enable-lora` to enable LoRA.\"," +
			`"loaded_adapters":null}`))
	})
})

var _ = Describe("sglang engine startup", func() {
	It("Should refuse to start with the KV cache enabled, naming the feature", func() {
		_, err := startServerWithArgs(context.TODO(),
			[]string{"cmd", "--engine", "sglang", "--model", common.TestModelName, "--enable-kvcache"})
		Expect(err).To(MatchError(ContainSubstring("KV-cache simulation is not implemented by the sglang engine yet")))
	})

	It("Should refuse a config file written for vLLM", func() {
		_, err := startServerWithArgs(context.TODO(),
			[]string{"cmd", "--engine", "sglang", "--config", "../../manifests/vllm-config.yaml"})
		Expect(err).To(MatchError(ContainSubstring("does not recognize the following configuration key(s)")))
	})
})
