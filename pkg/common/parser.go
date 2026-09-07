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
	// EngineEnv is read when the --engine flag is not passed; see configuration precedence in the docs.
	EngineEnv = "SIM_ENGINE"
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

// toggle sets a boolean pointer to a specific value when the flag is seen.
type toggle struct {
	ptr *bool
	val bool
}

// Set ignores the input and just applies the hardcoded boolean
func (t toggle) Set(_ string) error { *t.ptr = t.val; return nil }
func (t toggle) Type() string       { return "bool" }
func (t toggle) String() string     { return "" }

// AddToggle registers two distinct flags pointing to one variable
func AddToggle(f *pflag.FlagSet, ptr *bool, name, nameUsage, noNameUsage string) {
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

// ResolveEngineName determines which engine backend to run, following the same
// precedence as --model/--hash-seed: command-line flag > SIM_ENGINE env var >
// YAML config file > default ("vllm"). This must be resolved before
// ParseCommandParamsAndLoadConfig is called, since the caller uses it to select
// which engine's BindFlags/ValidateConfig to pass in.
func ResolveEngineName() (string, error) {
	if v := GetParamValueFromArgs("engine"); len(v) == 1 {
		return v[0], nil
	}
	if v := os.Getenv(EngineEnv); v != "" {
		return v, nil
	}
	if cf := GetParamValueFromArgs("config"); len(cf) == 1 {
		scratch := NewConfig()
		if err := scratch.load(cf[0]); err != nil {
			return "", err
		}
		return scratch.EngineName, nil
	}
	return "vllm", nil
}

// Engine supplies the active engine's own CLI flags and configuration
// validation, for use by ParseCommandParamsAndLoadConfig.
type Engine interface {
	// Name identifies the engine backend, e.g. "vllm".
	Name() string
	// BindFlags registers the engine's own CLI flags on f and reconciles any
	// values that need parsing beyond what pflag can bind directly. Must be
	// called before f.Parse.
	BindFlags(f *pflag.FlagSet, cfg *Configuration) error
	// ValidateConfig checks the engine's own fields of cfg. Called after cfg's
	// common fields have already been validated.
	ValidateConfig(cfg *Configuration) error
}

// ParseCommandParamsAndLoadConfig loads configuration, parses command line parameters, merges the values
// (command line overwrites the config file; see documentation for configuration precedence involving environment variables),
// and validates the configuration. eng is the already-resolved engine (see ResolveEngineName), and
// registers and validates that engine's own flags and fields.
func ParseCommandParamsAndLoadConfig(eng Engine) (*Configuration, error) {
	config := NewConfig()
	config.EngineName = eng.Name()

	configFileValues := GetParamValueFromArgs("config")
	if len(configFileValues) == 1 {
		if err := config.load(configFileValues[0]); err != nil {
			return nil, err
		}
	}

	servedModelNames := GetParamValueFromArgs("served-model-name")

	f := pflag.NewFlagSet("llm-d-inference-sim flags", pflag.ContinueOnError)

	f.IntVar(&config.Port, "port", config.Port, "Port")
	f.IntVar(&config.MaxRequestBodySizeMB, "max-request-body-size-mb", config.MaxRequestBodySizeMB, "Maximum allowed size of an HTTP request body in megabytes, must be between 1 and 512, default is 4 (matching the fasthttp built-in default)")
	f.StringVar(&config.Model, "model", config.Model,
		"Currently 'loaded' model (if omitted on the command line, "+ModelEnv+" may set the model; see docs)")
	f.IntVar(&config.MaxNumSeqs, "max-num-seqs", config.MaxNumSeqs, "Maximum number of inference requests that could be processed at the same time")
	f.IntVar(&config.MaxWaitingQueueLength, "max-waiting-queue-length", config.MaxWaitingQueueLength, "Maximum length of inference requests waiting queue")
	f.IntVar(&config.MaxModelLen, "max-model-len", config.MaxModelLen, "Model's context window, maximum number of tokens in a single request including input and output")

	f.StringVar(&config.Mode, "mode", config.Mode, "Simulator mode: echo - returns the same text that was sent in the request, for chat completion returns the last message; random - returns random sentence from a bank of pre-defined sentences")
	f.DurationVar(&config.InterTokenLatency, "inter-token-latency", config.InterTokenLatency, "Time to generate one token, e.g. 100ms")
	f.DurationVar(&config.TimeToFirstToken, "time-to-first-token", config.TimeToFirstToken, "Time to first token, e.g. 100ms")

	f.DurationVar(&config.PrefillOverhead, "prefill-overhead", config.PrefillOverhead, "Time to prefill, e.g. 100ms. This argument is ignored if <time-to-first-token> is not 0.")
	f.DurationVar(&config.PrefillTimePerToken, "prefill-time-per-token", config.PrefillTimePerToken, "Time to prefill per token, e.g. 100ms")
	f.DurationVar(&config.PrefillTimeStdDev, "prefill-time-std-dev", config.PrefillTimeStdDev, "Standard deviation for time to prefill, e.g. 100ms")

	f.DurationVar(&config.InterTokenLatencyStdDev, "inter-token-latency-std-dev", config.InterTokenLatencyStdDev, "Standard deviation for time between generated tokens, e.g. 100ms")
	f.DurationVar(&config.TimeToFirstTokenStdDev, "time-to-first-token-std-dev", config.TimeToFirstTokenStdDev, "Standard deviation for time before the first token will be returned, e.g. 100ms")
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
	f.IntVar(&config.ToolCallExtraCallProbability, "tool-call-extra-call-probability", config.ToolCallExtraCallProbability, "Probability (0-100) to make one additional tool call beyond the minimum; rolls repeat until a roll fails or all tools are called")

	f.IntVar(&config.DPSize, "data-parallel-size", config.DPSize, "Number of ranks to run")
	f.IntVar(&config.Rank, "data-parallel-rank", config.Rank, "The rank when running each rank in a process. If set, data-parallel-size is ignored")

	f.StringVar(&config.DatasetPath, "dataset-path", config.DatasetPath, "Local path to the sqlite db file for response generation from a dataset")
	f.StringVar(&config.DatasetURL, "dataset-url", config.DatasetURL, "URL to download the sqlite db file for response generation from a dataset")
	f.BoolVar(&config.DatasetInMemory, "dataset-in-memory", config.DatasetInMemory, "Load the entire dataset into memory for faster access")
	f.StringVar(&config.DatasetTableName, "dataset-table-name", config.DatasetTableName, "Table name for custom dataset, default is 'llmd'")

	f.StringVar(&config.RenderURL, "render-url", config.RenderURL, "URL of the tokenizer render service; when unset the simulated tokenizer is used")
	f.DurationVar(&config.RenderTimeout, "render-timeout", config.RenderTimeout, "Timeout for tokenizer render requests (e.g. 30s)")
	f.DurationVar(&config.MMRenderTimeout, "mm-render-timeout", config.MMRenderTimeout, "Timeout for multi-modal tokenizer render requests (e.g. 60s)")
	f.BoolVar(&config.ForceDummyTokenizer, "force-dummy-tokenizer", config.ForceDummyTokenizer, "(deprecated) Force the use of dummy tokenizer even if a real model name is provided; omit --render-url instead")

	f.DurationVar(&config.StartupDuration, "startup-duration", config.StartupDuration,
		"Duration to return 503 on /health/ready to simulate GPU loading (e.g. 30s). Default is 0 (immediately ready)")

	f.BoolVar(&config.EnableRequestIDHeaders, "enable-request-id-headers", config.EnableRequestIDHeaders, "Enable including X-Request-Id header in responses")
	f.BoolVar(&config.LogHTTP, "log-http", config.LogHTTP, "Log full HTTP request and response (method, URI, headers, bodies when buffered, status); streamed bodies are not logged")
	f.BoolVar(&config.SkipToolValidation, "skip-tool-validation", config.SkipToolValidation, "Skip the built-in validation of incoming tool schemas, matching real vLLM which forwards them to the model verbatim")

	f.IntVar(&config.FailureInjectionRate, "failure-injection-rate", config.FailureInjectionRate, "Probability (0-100) of injecting failures")
	failureTypes := GetParamValueFromArgs("failure-types")
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

	AddToggle(f, &config.Omni,
		"omni", "Enable omni mode: emit an image chunk when X-Send-Image header is present", "Disable omni mode")
	f.IntVar(&config.ImageEmissionRate, "image-emission-rate", config.ImageEmissionRate, "Probability (0-100) of emitting a synthetic image chunk per chat completion request in omni mode")
	f.DurationVar(&config.TimeToGenerateImage, "time-to-generate-image", config.TimeToGenerateImage, "Simulated time to generate an image in omni mode, e.g. 500ms")
	f.DurationVar(&config.TimeToGenerateImageStdDev, "time-to-generate-image-std-dev", config.TimeToGenerateImageStdDev, "Standard deviation for time to generate an image in omni mode, e.g. 50ms")

	// These values were manually parsed above in GetParamValueFromArgs, we leave this in order to get these flags in --help
	var dummyString string
	f.StringVar(&dummyString, "config", "", "The path to a yaml configuration file. The command line values overwrite the configuration file values")
	f.StringVar(&dummyString, "engine", "", "The inference engine backend to simulate (currently only 'vllm' is supported)")
	var dummyMultiString multiString
	f.Var(&dummyMultiString, "served-model-name", "Model names exposed by the API (a list of space-separated strings)")
	// In order to allow empty arguments, we set a dummy NoOptDefVal for these flags
	f.Lookup("served-model-name").NoOptDefVal = dummy
	f.Lookup("served-model-name").DefValue = ""

	if err := eng.BindFlags(f, config); err != nil {
		return nil, err
	}

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
	if servedModelNames != nil {
		config.ServedModelNames = servedModelNames
	}
	if failureTypes != nil {
		config.FailureTypes = failureTypes
	}

	// hash-seed is registered by the engine's BindFlags above, but its env-var
	// precedence is handled here alongside model's, on the same FlagSet.
	if !f.Changed("hash-seed") {
		if v := os.Getenv(PythonHashSeedEnv); v != "" {
			config.HashSeed = v
		}
	}

	if err := config.validate(); err != nil {
		return nil, err
	}
	if err := eng.ValidateConfig(config); err != nil {
		return nil, err
	}

	return config, nil
}

// GetParamValueFromArgs manually scans os.Args for a flag that takes multiple
// space-separated values (which pflag cannot bind directly to a slice the way
// this codebase needs), returning the values that followed it, if present.
func GetParamValueFromArgs(param string) []string {
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
