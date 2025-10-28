package llmdinferencesim

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/valyala/fasthttp/fasthttputil"
	"k8s.io/klog/v2"

	"github.com/onsi/gomega"
)

const (
	qwenModelName   = "Qwen/Qwen2-0.5B"
	baseURL         = "http://localhost/v1"
	testModel       = "testmodel"
	testUserMessage = "This is a test."
	metricsUrl      = "http://localhost/metrics"
)

var userMsgTokens int64

// Starts server in the given mode, no additional arguments or environment variables
func startServer(ctx context.Context, mode string) (*http.Client, error) {
	return startServerWithArgsAndEnv(ctx, mode, nil, nil)
}

// Starts server in the given mode and environment variables
func startServerWithEnv(ctx context.Context, mode string, envs map[string]string) (*http.Client, error) {
	return startServerWithArgsAndEnv(ctx, mode, nil, envs)
}

// Starts server according the given arguments
// if args are defined - mode defined in args will override the mode defined by the mode parameter
func startServerWithArgs(ctx context.Context, args []string) (*http.Client, error) {
	return startServerWithArgsAndEnv(ctx, "", args, nil)
}

// Starts server according the given parmaters: mode, arguments and environment
// if args are defined - the mode parameter is discarded, value from args is used
func startServerWithArgsAndEnv(ctx context.Context, mode string, args []string, envs map[string]string) (*http.Client, error) {
	oldArgs := os.Args
	defer func() {
		os.Args = oldArgs
	}()

	if args != nil {
		os.Args = args
	} else {
		os.Args = []string{"cmd", "--model", testModel, "--mode", mode}
	}

	if envs != nil {
		for k, v := range envs {
			err := os.Setenv(k, v)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		}

		defer func() {
			for k := range envs {
				err := os.Unsetenv(k)
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
			}
		}()
	}

	logger := klog.Background()

	s, err := New(logger)
	if err != nil {
		return nil, err
	}
	config, err := common.ParseCommandParamsAndLoadConfig()
	if err != nil {
		return nil, err
	}
	s.config = config

	// calculate number of tokens for user message,
	// must be activated after parseCommandParamsAndLoadConfig since it initializes the random engine
	userMsgTokens = int64(len(common.Tokenize(testUserMessage)))

	if err := s.initializeSim(ctx); err != nil {
		return nil, err
	}

	listener := fasthttputil.NewInmemoryListener()

	// start the http server
	go func() {
		if err := s.startServer(ctx, listener); err != nil {
			logger.Error(err, "error starting server")
		}
	}()

	return &http.Client{
		Transport: &http.Transport{
			DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
				return listener.Dial()
			},
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true,
			},
		},
	}, nil
}

// startServerAndSendRequest - starts server configured according the given latency parameters in echo mode,
// sends a single request with the given prompt
func startServerAndSendRequest(modelName string, prompt string, isStreaming bool, ttft int, prefillTimePerToken int, interTokenLatency int) *http.Client {
	ctx := context.TODO()
	args := []string{"cmd", "--model", modelName, "--mode", common.ModeEcho,
		// "--kv-cache-transfer-latency", strconv.Itoa(kvcacheTransferLatency),
		// "--kv-cache-transfer-time-per-token", strconv.Itoa(kvCacheTransferTimePerToken),
		"--time-to-first-token", strconv.Itoa(ttft),
		"--prefill-time-per-token", strconv.Itoa(prefillTimePerToken),
		"--inter-token-latency", strconv.Itoa(interTokenLatency),
	}

	client, err := startServerWithArgs(ctx, args)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	openaiclient, params := getOpenAIClientAndChatParams(client, modelName, prompt, isStreaming)

	// send a single request in a serial way
	_, err = openaiclient.Chat.Completions.New(ctx, params)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	return client
}

// sendSimpleChatRequest starts server using the given environment variables and sends one chat completions request
func sendSimpleChatRequest(envs map[string]string, streaming bool) *http.Response {
	ctx := context.TODO()

	client, err := startServerWithEnv(ctx, common.ModeRandom, envs)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	openaiclient, params := getOpenAIClientAndChatParams(client, testModel, testUserMessage, streaming)
	var httpResp *http.Response
	resp, err := openaiclient.Chat.Completions.New(ctx, params, option.WithResponseInto(&httpResp))
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(resp).NotTo(gomega.BeNil())

	gomega.Expect(resp.Choices).ShouldNot(gomega.BeEmpty())
	gomega.Expect(string(resp.Object)).To(gomega.Equal(chatCompletionObject))

	return httpResp
}

// getOpenAIClientAndChatParams - creates an openai client and params for /chat/completions call based on the given parameters
func getOpenAIClientAndChatParams(client option.HTTPClient, model string, message string,
	streaming bool) (openai.Client, openai.ChatCompletionNewParams) {
	openaiclient := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithHTTPClient(client))

	params := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(message),
		},
		Model: model,
	}
	if streaming {
		params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
	}
	return openaiclient, params
}

// nolint
// getOpenAIClentAndCompletionParams - creates an openai client and params for /completions call based on the given parameters
func getOpenAIClentAndCompletionParams(client option.HTTPClient, model string, message string,
	streaming bool) (openai.Client, openai.CompletionNewParams) {
	openaiclient := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithHTTPClient(client))

	params := openai.CompletionNewParams{
		Prompt: openai.CompletionNewParamsPromptUnion{
			OfString: openai.String(message),
		},
		Model: openai.CompletionNewParamsModel(model),
	}
	if streaming {
		params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: param.NewOpt(true)}
	}
	return openaiclient, params
}

// isLoraMetricPresent checks if a matching metric exists
// metrics: the list of metrics
// running: list of loras in running_lora_adapters, the order does not matter
// waiting: list of loras in waiting_lora_adapters, the order does not matter
func isLoraMetricPresent(metrics []string, running, waiting []string) bool {
	return findLoraMetric(metrics, running, waiting) != ""
}

// getLoraTimestamp returns timestamp or nil, error
func getLoraTimestamp(metrics []string, running, waiting []string) (*float64, error) {
	metric := findLoraMetric(metrics, running, waiting)
	if metric == "" {
		return nil, nil // not found
	}
	return extractTimestamp(metric)
}

// extractTimestamp gets timestamp from the given metric
func extractTimestamp(metric string) (*float64, error) {
	// Extract timestamp: last part after space
	parts := strings.Split(metric, " ")
	if len(parts) < 2 {
		return nil, errors.New("invalid metric format")
	}
	timestampStr := parts[len(parts)-1]
	timestamp, err := strconv.ParseFloat(timestampStr, 64)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	return &timestamp, nil
}

func getLoraValidTimestamp(metrics []string, running, waiting []string) float64 {
	timestamp, err := getLoraTimestamp(metrics, running, waiting)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(timestamp).ToNot(gomega.BeNil())
	return *timestamp
}

func getLastLoraMetrics(metrics []string) ([]string, error) {
	lastTimestamp := float64(0)
	var lastMetrics []string
	for _, metric := range metrics {
		if strings.HasPrefix(metric, "vllm:lora_requests_info") {
			timestamp, err := extractTimestamp(metric)
			if err != nil {
				return nil, err
			}
			if lastTimestamp > *timestamp {
				continue
			}
			lastTimestamp = *timestamp
			if lastTimestamp < *timestamp {
				lastMetrics = make([]string, 0)
			}
			lastMetrics = append(lastMetrics, metric)
		}
	}
	return lastMetrics, nil
}

// findLoraMetric finds the relevant metric by comparing with the given loras sets (ignoring order)
// metrics: lines of metrics
// running: list of running loras to find
// waiting: list of waiting loras to find
// Looks for a line with the given running and waiting loras sets, the comparison is order agnostic.
// Return metric should match in both running and waiting sets.
// E.g. for input running=["l1", "l2", "l3"] and waiting=[] will return metric
// with running_lora_adapters=["l3", "l1", "l2"] and waiting_lora_adapters=[]
func findLoraMetric(metrics []string, running, waiting []string) string {
	// sort input arrays before compare, create string of all values, separated by comma
	sort.Strings(running)
	sort.Strings(waiting)
	runStr := strings.Join(running, ",")
	waitStr := strings.Join(waiting, ",")

	// regex to extract lora metrics and values
	re := regexp.MustCompile(`vllm:lora_requests_info\{.*running_lora_adapters="([^"]*)".*waiting_lora_adapters="([^"]*)".*\}\s+([0-9.e\+\-]+)`)
	for _, metric := range metrics {
		matches := re.FindStringSubmatch(metric)
		if len(matches) == 4 {
			// this line contains loraInfo metric, check running and waiting loras lists
			// split and sort metric's running and waiting loras lists for the comparison
			metricRun := splitString(matches[1])
			metricWait := splitString(matches[2])
			sort.Strings(metricRun)
			sort.Strings(metricWait)
			// if both lists are the same - return the metric
			if strings.Join(metricRun, ",") == runStr && strings.Join(metricWait, ",") == waitStr {
				return metric
			}
		} // if the metric is not in the required format - skip it
	}

	// required metric was not found
	return ""
}

// splits the given string to array of strings with separator = ","
func splitString(str string) []string {
	if str == "" {
		return []string{}
	}
	return strings.Split(str, ",")
}

// findMetric returns the value for the first metrics with the given prefix or an empty string if not found
func findMetric(metrics []string, metricPrefix string) string {
	// regex to extract metrics and values
	for _, metric := range metrics {
		if strings.Contains(metric, metricPrefix) {
			arr := strings.Split(metric, " ")
			if len(arr) == 2 {
				return arr[1]
			}
			break
		}
	}
	// required metric was not found
	return ""
}

// findIntMetric returns the value for the first metrics with the given prefix as int or nil if not found
func findIntMetric(metrics []string, metricPrefix string) *int {
	valueStr := findMetric(metrics, metricPrefix)
	if valueStr == "" {
		return nil
	}

	val, err := strconv.Atoi(valueStr)
	if err != nil {
		return nil
	}
	return &val
}

// getFloatBucketMetricLine builds a string which will defin bucket metric line for the given parameters
// model the model name
// metrics the metric name
// bucketBoundary the upper bucket boundary, Inf(1) defines the last bucket
// count bucket samples count
func getFloatBucketMetricLine(model string, metric string, bucketBoundary float64, count int) string {
	return fmt.Sprintf("%s %d", getFloatBucketMetricPrefix(model, metric, bucketBoundary), count)
}

// same as getFloatBucketMetricLine but without the value part
func getFloatBucketMetricPrefix(model string, metric string, bucketBoundary float64) string {
	buckerBoundStr := "+Inf"
	if bucketBoundary != math.Inf(1) {
		buckerBoundStr = fmt.Sprintf("%g", bucketBoundary)
	}
	return fmt.Sprintf("%s_bucket{model_name=\"%s\",le=\"%s\"}", metric, model, buckerBoundStr)
}

// checkBucketBoundary checks that the given bucket's samples count is valid according the given parameters
// Scenario is a single request, so buckets counts could be 0 or 1.
// Buckets lower than the expected value should have count 0, other buckets - count 1.
// Important note: since metrics represent real timing, it could be a little bit higher than the expected,
// which is based on the pure latencies calculations, on in case the expected value is equal or very close to the
// upper bounary we can get any value (0 or 1), in this case we don't check this bucket
// metrics the full metrics response
// modelName the model name
// metricName the specific metric name
// bucketBoudary the upper boundary of the required bucket
// prevBoundary the upper boundary of the previous bucket
// expectedValue expected value in the histogram
func checkBucketBoundary(metrics string, modelName string, metricName string, bucketBoudary float64,
	prevBoundary float64, expectedValue float64) {
	if expectedValue > prevBoundary && bucketBoudary >= expectedValue && (bucketBoudary-expectedValue) < 0.005 {
		// expected time is too close to the bucket's boudary
		// it's possiblt that in theory we expect 1 in this bucket but will get 0 and this situation is ok
		// since there is some additional calculation time
		fmt.Printf("Expected value is too close to the boundary - skip test for this bucket (%.4f - %.4f] and expected value %.4f\n",
			prevBoundary, bucketBoudary, expectedValue)
		return
	}
	expectedCount := 0
	if bucketBoudary > expectedValue {
		expectedCount = 1
	}
	gomega.Expect(metrics).To(gomega.ContainSubstring(getFloatBucketMetricLine(modelName, metricName, bucketBoudary, expectedCount)))
}

// checkLatencyMertics sends /metrics request and checks that latency related values are valid
// client the http client to be used for request send
// modelName the model name
// numOfOutputTokens number of tokens in the output of the completion request we want to validate
// ttft time to first token parameter
// prefillTimePerToken prefill time per input tokens
// interTokenLatency processing time per output token
func checkLatencyMertics(client *http.Client, modelName string, numOfInputTokens int, numOfOutputTokens int, ttft int, prefillTimePerToken int, interTokenLatency int) {
	// wait a little bit and check metrics
	time.Sleep(300 * time.Millisecond)
	metricsResp, err := client.Get(metricsUrl)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(metricsResp.StatusCode).To(gomega.Equal(http.StatusOK))

	data, err := io.ReadAll(metricsResp.Body)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	metrics := string(data)

	var expectedPrefillTime float64
	// TODO take into consideration remote prefill
	if ttft > 0 {
		// time-to-first-token overwrites calculation of prefill time based on number of input tokens
		expectedPrefillTime = float64(ttft) / 1000

	} else {
		expectedPrefillTime = float64(numOfInputTokens*prefillTimePerToken) / 1000
	}
	expectedDecodeTime := float64(interTokenLatency*(numOfOutputTokens-1)) / 1000
	expectedE2ELatency := expectedPrefillTime + expectedDecodeTime

	prevBoundary := math.Inf(-1)

	for _, bucketBoudary := range common.RequestLatencyBucketsBoundaries {
		checkBucketBoundary(metrics, modelName, prefillTimeMetricName, bucketBoudary, prevBoundary, expectedPrefillTime)
		checkBucketBoundary(metrics, modelName, decodeTimeMetricName, bucketBoudary, prevBoundary, expectedDecodeTime)
		checkBucketBoundary(metrics, modelName, e2eReqLatencyMetricName, bucketBoudary, prevBoundary, expectedE2ELatency)

		prevBoundary = bucketBoudary
	}
	// check the last bucket
	lastBoundary := common.RequestLatencyBucketsBoundaries[len(common.RequestLatencyBucketsBoundaries)-1]
	checkBucketBoundary(metrics, modelName, prefillTimeMetricName, math.Inf(1), lastBoundary, expectedPrefillTime)
	checkBucketBoundary(metrics, modelName, decodeTimeMetricName, math.Inf(1), lastBoundary, expectedDecodeTime)
	checkBucketBoundary(metrics, modelName, e2eReqLatencyMetricName, math.Inf(1), lastBoundary, expectedE2ELatency)
}
