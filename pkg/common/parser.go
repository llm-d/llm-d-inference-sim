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

package common

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/pflag"
	"k8s.io/klog/v2"
)

const (
	dummy                = " "
	vllmServerDevModeEnv = "VLLM_SERVER_DEV_MODE"
	PodNameEnv           = "POD_NAME"
	PodNsEnv             = "POD_NAMESPACE"
	// ModelEnv is read when the --model flag is not passed; see configuration precedence in the docs.
	ModelEnv = "SIM_MODEL"
	// PythonHashSeedEnv is read when the --hash-seed flag is not passed; see configuration precedence in the docs.
	PythonHashSeedEnv = "PYTHONHASHSEED"
)

// Needed to parse values that contain multiple strings
type multiString struct {
	values []string
}

func (l *multiString) String() string {
	return strings.Join(l.values, " ")
}

func (l *multiString) Set(val string) error {
	l.values = append(l.values, val)
	return nil
}

func (l *multiString) Type() string {
	return "strings"
}

func (c *Configuration) unmarshalLoras() error {
	c.LoraModules = make([]LoraModule, 0)
	for _, jsonStr := range c.LoraModulesString {
		var lora LoraModule
		if err := json.Unmarshal([]byte(jsonStr), &lora); err != nil {
			return err
		}
		c.LoraModules = append(c.LoraModules, lora)
	}
	return nil
}

// toggle sets a boolean pointer to a specific value when the flag is seen.
type toggle struct {
	ptr *bool
	val bool
}

// Set ignores the input and just applies the hardcoded boolean
func (t toggle) Set(_ string) error { *t.ptr = t.val; return nil }
func (t toggle) Type() string       { return "bool" }
func (t toggle) String() string     { return "" }

// addToggle registers two distinct flags pointing to one variable
func addToggle(f *pflag.FlagSet, ptr *bool, name, nameUsage, noNameUsage string) {
	// Register Positive Flag
	f.Var(toggle{ptr, true}, name, nameUsage)
	f.Lookup(name).NoOptDefVal = "true"
	f.Lookup(name).DefValue = "" // Hides the [=t] in help

	// Register Negative Flag
	noName := "no-" + name
	f.Var(toggle{ptr, false}, noName, noNameUsage)
	f.Lookup(noName).NoOptDefVal = "true"
	f.Lookup(noName).DefValue = "" // Hides the [=t] in help
}

// ParseCommandParamsAndLoadConfig loads configuration, parses command line parameters, merges the values
// (command line overwrites the config file; see documentation for configuration precedence involving environment variables),
// and validates the configuration.
func ParseCommandParamsAndLoadConfig() (*Configuration, error) {
	config := newConfig()

	configFileValues := getParamValueFromArgs("config")
	if len(configFileValues) == 1 {
		if err := config.load(configFileValues[0]); err != nil {
			return nil, err
		}
	}

	servedModelNames := getParamValueFromArgs("served-model-name")
	loraModuleNames := getParamValueFromArgs("lora-modules")
	fakeMetrics := getParamValueFromArgs("fake-metrics")

	f := pflag.NewFlagSet("llm-d-inference-sim flags", pflag.ContinueOnError)

	f.IntVar(&config.Port, "port", config.Port, "Port")
	f.StringVar(&config.Model, "model", config.Model,
		"Currently 'loaded' model (if omitted on the command line, "+ModelEnv+" may set the model; see docs)")
	f.IntVar(&config.MaxNumSeqs, "max-num-seqs", config.MaxNumSeqs, "Maximum number of inference requests that could be processed at the same time")
	f.IntVar(&config.MaxWaitingQueueLength, "max-waiting-queue-length", config.MaxWaitingQueueLength, "Maximum length of inference requests waiting queue")
	f.IntVar(&config.MaxLoras, "max-loras", config.MaxLoras, "Maximum number of LoRAs in a single batch")
	f.IntVar(&config.MaxCPULoras, "max-cpu-loras", config.MaxCPULoras, "Maximum number of LoRAs to store in CPU memory")
	f.IntVar(&config.MaxModelLen, "max-model-len", config.MaxModelLen, "Model's context window, maximum number of tokens in a single request including input and output")

	f.StringVar(&config.Mode, "mode", config.Mode, "Simulator mode: echo - returns the same text that was sent in the request, for chat completion returns the last message; random - returns random sentence from a bank of pre-defined sentences")
	f.Var(&config.InterTokenLatency, "inter-token-latency", "Time to generate one token (e.g. 100ms. Integer format is deprecated)")
	f.Var(&config.TimeToFirstToken, "time-to-first-token", "Time to first token (e.g. 100ms. Integer format is deprecated)")

	f.Var(&config.PrefillOverhead, "prefill-overhead", "Time to prefill (e.g. 100ms. Integer format is deprecated). This argument is ignored if <time-to-first-token> is not 0.")
	f.Var(&config.PrefillTimePerToken, "prefill-time-per-token", "Time to prefill per token (e.g. 100ms. Integer format is deprecated)")
	f.Var(&config.PrefillTimeStdDev, "prefill-time-std-dev", "Standard deviation for time to prefill (e.g. 100ms. Integer format is deprecated)")
	f.Var(&config.KVCacheTransferTimePerToken, "kv-cache-transfer-time-per-token", "Time for KV-cache transfer per token from a remote vLLM (e.g. 100ms. Integer format is deprecated)")
	f.Var(&config.KVCacheTransferTimeStdDev, "kv-cache-transfer-time-std-dev", "Standard deviation for time for KV-cache transfer per token from a remote vLLM (e.g. 100ms. Integer format is deprecated)")

	f.Var(&config.KVCacheTransferLatency, "kv-cache-transfer-latency", "Time for KV-cache transfer from a remote vLLM (e.g. 100ms. Integer format is deprecated)")
	f.Var(&config.InterTokenLatencyStdDev, "inter-token-latency-std-dev", "Standard deviation for time between generated tokens (e.g. 100ms. Integer format is deprecated)")
	f.Var(&config.TimeToFirstTokenStdDev, "time-to-first-token-std-dev", "Standard deviation for time before the first token will be returned (e.g. 100ms. Integer format is deprecated)")
	f.Var(&config.KVCacheTransferLatencyStdDev, "kv-cache-transfer-latency-std-dev", "Standard deviation for time for KV-cache transfer from a remote vLLM (e.g. 100ms. Integer format is deprecated)")
	f.Int64Var(&config.Seed, "seed", config.Seed, "Random seed for operations (if not set, current Unix time in nanoseconds is used)")
	f.Float64Var(&config.TimeFactorUnderLoad, "time-factor-under-load", config.TimeFactorUnderLoad, "Time factor under load (must be >= 1.0)")

	f.IntVar(&config.MaxToolCallIntegerParam, "max-tool-call-integer-param", config.MaxToolCallIntegerParam, "Maximum possible value of integer parameters in a tool call")
	f.IntVar(&config.MinToolCallIntegerParam, "min-tool-call-integer-param", config.MinToolCallIntegerParam, "Minimum possible value of integer parameters in a tool call")
	f.Float64Var(&config.MaxToolCallNumberParam, "max-tool-call-number-param", config.MaxToolCallNumberParam, "Maximum possible value of number (float) parameters in a tool call")
	f.Float64Var(&config.MinToolCallNumberParam, "min-tool-call-number-param", config.MinToolCallNumberParam, "Minimum possible value of number (float) parameters in a tool call")
	f.IntVar(&config.MaxToolCallArrayParamLength, "max-tool-call-array-param-length", config.MaxToolCallArrayParamLength, "Maximum possible length of array parameters in a tool call")
	f.IntVar(&config.MinToolCallArrayParamLength, "min-tool-call-array-param-length", config.MinToolCallArrayParamLength, "Minimum possible length of array parameters in a tool call")
	f.IntVar(&config.ToolCallNotRequiredParamProbability, "tool-call-not-required-param-probability", config.ToolCallNotRequiredParamProbability, "Probability to add a parameter, that is not required, in a tool call")
	f.IntVar(&config.ObjectToolCallNotRequiredParamProbability, "object-tool-call-not-required-field-probability", config.ObjectToolCallNotRequiredParamProbability, "Probability to add a field, that is not required, in an object in a tool call")

	f.BoolVar(&config.EnableKVCache, "enable-kvcache", config.EnableKVCache, "Defines if KV cache feature is enabled")
	f.IntVar(&config.KVCacheSize, "kv-cache-size", config.KVCacheSize, "Maximum number of token blocks in kv cache")
	f.Float64Var(&config.GlobalCacheHitThreshold, "global-cache-hit-threshold", 0, "Default cache hit threshold [0, 1] for all requests. If a request specifies cache_hit_threshold, it takes precedence")
	f.IntVar(&config.TokenBlockSize, "block-size", config.TokenBlockSize, "Token block size for contiguous chunks of tokens, possible values: 8,16,32,64,128")
	f.StringVar(&config.HashSeed, "hash-seed", config.HashSeed,
		"Seed for hash generation (if omitted on the command line, "+PythonHashSeedEnv+" may set it; see docs)")
	f.StringVar(&config.ZMQEndpoint, "zmq-endpoint", config.ZMQEndpoint, "ZMQ address to publish events")
	f.IntVar(&config.EventBatchSize, "event-batch-size", config.EventBatchSize, "Maximum number of kv-cache events to be sent together")
	f.IntVar(&config.DPSize, "data-parallel-size", config.DPSize, "Number of ranks to run")
	f.IntVar(&config.Rank, "data-parallel-rank", config.Rank, "The rank when running each rank in a process. If set, data-parallel-size is ignored")

	f.StringVar(&config.DatasetPath, "dataset-path", config.DatasetPath, "Local path to the sqlite db file for response generation from a dataset")
	f.StringVar(&config.DatasetURL, "dataset-url", config.DatasetURL, "URL to download the sqlite db file for response generation from a dataset")
	f.BoolVar(&config.DatasetInMemory, "dataset-in-memory", config.DatasetInMemory, "Load the entire dataset into memory for faster access")
	f.StringVar(&config.DatasetTableName, "dataset-table-name", config.DatasetTableName, "Table name for custom dataset, default is 'llmd'")

	f.StringVar(&config.UDSSocketPath, "uds-socket-path", config.UDSSocketPath, "UDS socket path for communication with HF tokenizer, default is '/tmp/tokenizer/tokenizer-uds.socket'")

	f.DurationVar(&config.StartupDuration, "startup-duration", config.StartupDuration,
		"Duration to return 503 on /health/ready to simulate GPU loading (e.g. 30s). Default is 0 (immediately ready)")

	addToggle(f, &config.EnableSleepMode, "enable-sleep-mode", "Enable sleep mode", "Disable sleep mode")
	f.BoolVar(&config.EnableRequestIDHeaders, "enable-request-id-headers", config.EnableRequestIDHeaders, "Enable including X-Request-Id header in responses")

	f.IntVar(&config.FailureInjectionRate, "failure-injection-rate", config.FailureInjectionRate, "Probability (0-100) of injecting failures")
	failureTypes := getParamValueFromArgs("failure-types")
	var dummyFailureTypes multiString
	failureTypesDescription := fmt.Sprintf("List of specific failure types to inject (%s, %s, %s, %s, %s, %s)",
		FailureTypeRateLimit, FailureTypeInvalidAPIKey, FailureTypeContextLength, FailureTypeServerError, FailureTypeInvalidRequest,
		FailureTypeModelNotFound)
	f.Var(&dummyFailureTypes, "failure-types", failureTypesDescription)
	f.Lookup("failure-types").NoOptDefVal = dummy
	f.Lookup("failure-types").DefValue = ""

	f.StringVar(&config.SSLCertFile, "ssl-certfile", config.SSLCertFile, "Path to SSL certificate file for HTTPS (optional)")
	f.StringVar(&config.SSLKeyFile, "ssl-keyfile", config.SSLKeyFile, "Path to SSL private key file for HTTPS (optional)")
	f.BoolVar(&config.SelfSignedCerts, "self-signed-certs", config.SelfSignedCerts, "Enable automatic generation of self-signed certificates for HTTPS")

	f.StringVar(&config.LatencyCalculator, "latency-calculator", config.LatencyCalculator,
		`Name of the latency calculator to be used in the response generation (optional). The default calculation is based on the current load of the simulator and on 
		the configured latency parameters, e.g., time-to-first-token and prefill-time-per-token`)

	f.IntVar(&config.DefaultEmbeddingDimensions, "default-embedding-dimensions", config.DefaultEmbeddingDimensions,
		"Default size of embedding vectors when the request does not specify dimensions (used by /v1/embeddings)")

	f.DurationVar(&config.FakeMetricsRefreshInterval, "fake-metrics-refresh-interval", config.FakeMetricsRefreshInterval,
		"Defines how often function-based fake metrics are recalculated, defaults to 100ms")

	addToggle(f, &config.MMEncoderOnly,
		"mm-encoder-only", "Skip the language component of the model", "Don't skip the language component of the model")
	f.StringVar(&config.MMProcessorKWArgs, "mm-processor-kwargs", config.MMProcessorKWArgs, "Arguments to be forwarded to the model's processor for multi-modal data, ignored")
	f.StringVar(&config.ECTransferConfig, "ec-transfer-config", config.ECTransferConfig, "Configuration for distributed EC cache transfer, ignored")
	addToggle(f, &config.EnforceEager,
		"enforce-eager", "Always use eager-mode PyTorch, ignored", "Don't always use eager-mode PyTorch, ignored")
	addToggle(f, &config.EnablePrefixCaching,
		"enable-prefix-caching", "Enable prefix caching, ignored", "Disable prefix caching, ignored")

	// These values were manually parsed above in getParamValueFromArgs, we leave this in order to get these flags in --help
	var dummyString string
	f.StringVar(&dummyString, "config", "", "The path to a yaml configuration file. The command line values overwrite the configuration file values")
	var dummyMultiString multiString
	f.Var(&dummyMultiString, "served-model-name", "Model names exposed by the API (a list of space-separated strings)")
	f.Var(&dummyMultiString, "lora-modules", "List of LoRA adapters (a list of space-separated JSON strings)")
	f.Var(&dummyMultiString, "fake-metrics", "A set of metrics to report to Prometheus instead of the real metrics")
	// In order to allow empty arguments, we set a dummy NoOptDefVal for these flags
	f.Lookup("served-model-name").NoOptDefVal = dummy
	f.Lookup("served-model-name").DefValue = ""
	f.Lookup("lora-modules").NoOptDefVal = dummy
	f.Lookup("lora-modules").DefValue = ""
	f.Lookup("fake-metrics").NoOptDefVal = dummy
	f.Lookup("fake-metrics").DefValue = ""

	flagSet := flag.NewFlagSet("simFlagSet", flag.ExitOnError)
	klog.InitFlags(flagSet)
	f.AddGoFlagSet(flagSet)

	// set default value for logger verbosity to INFO
	if err := flagSet.Set("v", "2"); err != nil {
		return nil, err
	}

	if err := f.Parse(os.Args[1:]); err != nil {
		if err == pflag.ErrHelp {
			// --help - exit without printing an error message
			os.Exit(0)
		}
		return nil, err
	}

	// Set the values for Pod Name, Pod Namespace and the VLLM Dev mode
	config.PodName = os.Getenv(PodNameEnv)
	config.PodNameSpace = os.Getenv(PodNsEnv)
	config.VllmDevMode = os.Getenv(vllmServerDevModeEnv) == "1"

	// Precedence for model and hash-seed: command-line flags > these env vars > YAML > defaults.
	if !f.Changed("model") {
		if v := os.Getenv(ModelEnv); v != "" {
			config.Model = v
		}
	}

	// Need to read in a variable to avoid merging the values with the config file ones
	if loraModuleNames != nil {
		config.LoraModulesString = loraModuleNames
		if err := config.unmarshalLoras(); err != nil {
			return nil, err
		}
	}
	if fakeMetrics != nil {
		if err := config.unmarshalFakeMetrics(fakeMetrics[0]); err != nil {
			return nil, err
		}
	}
	if servedModelNames != nil {
		config.ServedModelNames = servedModelNames
	}
	if failureTypes != nil {
		config.FailureTypes = failureTypes
	}

	if !f.Changed("hash-seed") {
		if v := os.Getenv(PythonHashSeedEnv); v != "" {
			config.HashSeed = v
		}
	}

	if err := config.validate(); err != nil {
		return nil, err
	}

	return config, nil
}

func getParamValueFromArgs(param string) []string {
	var values []string
	var readValues bool
	for _, arg := range os.Args[1:] {
		if readValues {
			if strings.HasPrefix(arg, "--") {
				break
			}
			if arg != "" {
				values = append(values, arg)
			}
		} else {
			if arg == "--"+param {
				readValues = true
				values = make([]string, 0)
			} else if strings.HasPrefix(arg, "--"+param+"=") {
				// Handle --param=value
				values = append(values, strings.TrimPrefix(arg, "--"+param+"="))
				break
			}
		}
	}

	return values
}
