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
	"strconv"
	"strings"

	"github.com/spf13/pflag"
	"k8s.io/klog/v2"
)

const (
	dummy = " "
	// boolNoOptDefVal is the NoOptDefVal AddToggle gives both spellings of a
	// toggle, so a bare "--flag" sets it without consuming a following value.
	boolNoOptDefVal = "true"
	PodNameEnv      = "POD_NAME"
	PodNsEnv        = "POD_NAMESPACE"
	// ModelEnv is read when the --model flag is not passed; see configuration precedence in the docs.
	ModelEnv = "SIM_MODEL"
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

// toggle sets a boolean pointer when the flag is seen. Both spellings of a
// setting share one variable (see AddToggle); val carries which spelling this
// flag is, so the negative one stores the negation of what it parses.
type toggle struct {
	ptr *bool
	val bool
}

// Set applies val for the bare "--flag" form, which NoOptDefVal turns into
// "true". A value given as "--flag=false" is honored rather than discarded, so
// the two spellings stay each other's negation: --flag=false means --no-flag.
func (t toggle) Set(s string) error {
	v, err := strconv.ParseBool(s)
	if err != nil {
		return fmt.Errorf("invalid boolean value %q", s)
	}
	*t.ptr = t.val == v
	return nil
}
func (t toggle) Type() string   { return "bool" }
func (t toggle) String() string { return "" }

// rejectSeparateBoolValue refuses "--flag <value>" for a boolean flag. pflag
// never consumes the following argument for one, since its NoOptDefVal already
// supplies the value, so the flag is set and the value is left behind as a
// positional that nothing reads: "--flag false" silently means "--flag".
//
// Real vLLM rejects the form too, since the value lands on the positional model
// argument that "vllm serve <model>" has already filled, so accepting it would
// diverge even where it happens to be harmless.
//
// Any following argument counts, not just a falsehood, because "--flag true" is
// equally unfaithful and reads as though the value were doing something. Only
// the argument directly after the flag is examined: the flags taking several
// space-separated values (e.g. served-model-name, failure-types) leave their own
// values as positionals, and those follow their own flag rather than a boolean.
func rejectSeparateBoolValue(f *pflag.FlagSet, args []string) error {
	for i, arg := range args {
		if i+1 >= len(args) || strings.HasPrefix(args[i+1], "-") {
			continue
		}
		name, isLong := strings.CutPrefix(arg, "--")
		if !isLong || strings.Contains(name, "=") {
			continue
		}
		if flag := f.Lookup(name); flag != nil && flag.Value.Type() == "bool" {
			// The negation of a "--no-" flag is the flag itself, so suggesting
			// another "--no-" prefix would name a flag that does not exist.
			opposite := "no-" + name
			if positive, ok := strings.CutPrefix(name, "no-"); ok {
				opposite = positive
			}
			return fmt.Errorf("--%s does not take a value: write \"--%s\" on its own, "+
				"or %q for the opposite setting", name, name, "--"+opposite)
		}
	}
	return nil
}

// rejectUnexpectedArgs refuses any raw argument pflag left unconsumed that is
// not one of the several space-separated values a flag like served-model-name
// or lora-modules deliberately leaves behind for GetParamValueFromArgs to
// collect. This tool takes no positional arguments (model is set with
// --model, not with a positional the way "vllm serve <model>" takes one), so
// anything else left over is a mistake: a value stranded next to a flag
// written as "--flag=value", a typo, or a leftover from editing a command
// line, rather than something to silently ignore.
//
// f.Parse has already succeeded by the time this runs, so every "--name"
// token here names a registered flag; how many of the following raw
// arguments it consumes follows from its NoOptDefVal, the same field that
// drives pflag's own parsing: "true" for a boolean-style flag (none), the
// dummy sentinel for the several-values flags (every following argument up
// to the next "--flag"), and anything else for an ordinary flag (exactly
// one).
func rejectUnexpectedArgs(f *pflag.FlagSet, args []string) error {
	var stray []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		rest, isLong := strings.CutPrefix(arg, "--")
		if !isLong {
			if arg != "" && !strings.HasPrefix(arg, "-") {
				stray = append(stray, arg)
			}
			continue
		}
		name, _, hasInlineValue := strings.Cut(rest, "=")
		flag := f.Lookup(name)
		if flag == nil || hasInlineValue {
			continue
		}
		switch flag.NoOptDefVal {
		case boolNoOptDefVal:
			// boolean-style flag: takes no value
		case dummy:
			for i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
				i++
			}
		default:
			// ordinary flag: consumes exactly one following value
			i++
		}
	}
	if len(stray) == 0 {
		return nil
	}
	return fmt.Errorf("unrecognized arguments: %s", strings.Join(stray, " "))
}

// AddToggle registers two distinct flags pointing to one variable
func AddToggle(f *pflag.FlagSet, ptr *bool, name, nameUsage, noNameUsage string) {
	// Register Positive Flag
	f.Var(toggle{ptr, true}, name, nameUsage)
	f.Lookup(name).NoOptDefVal = boolNoOptDefVal
	f.Lookup(name).DefValue = "" // Hides the [=t] in help

	// Register Negative Flag
	noName := "no-" + name
	f.Var(toggle{ptr, false}, noName, noNameUsage)
	f.Lookup(noName).NoOptDefVal = boolNoOptDefVal
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
		if _, err := scratch.load(cf[0]); err != nil {
			return "", err
		}
		return scratch.EngineName, nil
	}
	return DefaultEngineName, nil
}

// Engine supplies the active engine's own defaults, CLI flags, and
// configuration validation, for use by ParseCommandParamsAndLoadConfig.
type Engine interface {
	// Name identifies the engine backend, e.g. "vllm".
	Name() string
	// ApplyDefaults fills in the default values of the configuration groups
	// the engine owns. Called on a freshly constructed Configuration, before
	// a config file is loaded and before BindFlags, so that a YAML value
	// overrides a default and a flag overrides both.
	ApplyDefaults(cfg *Configuration)
	// BindFlags registers the engine's own CLI flags on f and reconciles any
	// values that need parsing beyond what pflag can bind directly, including
	// its own engine-specific groups (e.g. lora) from rawYAML, the raw YAML
	// tree returned by Configuration.load (nil if no --config file was
	// given). The engine must delete each group it consumes from rawYAML:
	// whatever is left once this returns is reported as an unrecognized
	// configuration key. Must be called before f.Parse.
	BindFlags(f *pflag.FlagSet, cfg *Configuration, rawYAML map[string]any) error
	// ApplyEnv applies the engine's own environment-variable settings to cfg.
	// Called after the flags have been parsed and before validation; changed
	// reports whether a given flag was set on the command line, so that an
	// env var can act as a fallback rather than an override.
	ApplyEnv(cfg *Configuration, changed func(flag string) bool)
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
	eng.ApplyDefaults(config)

	var rawYAML map[string]any
	configFileValues := GetParamValueFromArgs("config")
	if len(configFileValues) == 1 {
		var err error
		rawYAML, err = config.load(configFileValues[0])
		if err != nil {
			return nil, err
		}
	}
	// Set after the config file is loaded: its "engine" key maps onto
	// EngineName, but eng was resolved from the full precedence chain, so a flag
	// or SIM_ENGINE naming a different engine must not be undone here.
	config.EngineName = eng.Name()

	servedModelNames := GetParamValueFromArgs("served-model-name")

	f := pflag.NewFlagSet("llm-d-inference-sim flags", pflag.ContinueOnError)

	f.IntVar(&config.Port, "port", config.Port, "Port")
	f.IntVar(&config.MaxRequestBodySizeMB, "max-request-body-size-mb", config.MaxRequestBodySizeMB, "Maximum allowed size of an HTTP request body in megabytes, must be between 1 and 512, default is 4 (matching the fasthttp built-in default)")
	f.StringVar(&config.Model, "model", config.Model,
		"Currently 'loaded' model (if omitted on the command line, "+ModelEnv+" may set the model; see docs)")
	f.StringVar(&config.Mode, "mode", config.Mode, "Simulator mode: echo - returns the same text that was sent in the request, for chat completion returns the last message; random - returns random sentence from a bank of pre-defined sentences")
	f.DurationVar(&config.Latencies.InterTokenLatency, "inter-token-latency", config.Latencies.InterTokenLatency, "Time to generate one token, e.g. 100ms")
	f.DurationVar(&config.Latencies.TimeToFirstToken, "time-to-first-token", config.Latencies.TimeToFirstToken, "Time to first token, e.g. 100ms")

	f.DurationVar(&config.Latencies.PrefillOverhead, "prefill-overhead", config.Latencies.PrefillOverhead, "Time to prefill, e.g. 100ms. This argument is ignored if <time-to-first-token> is not 0.")
	f.DurationVar(&config.Latencies.PrefillTimePerToken, "prefill-time-per-token", config.Latencies.PrefillTimePerToken, "Time to prefill per token, e.g. 100ms")
	f.DurationVar(&config.Latencies.PrefillTimeStdDev, "prefill-time-std-dev", config.Latencies.PrefillTimeStdDev, "Standard deviation for time to prefill, e.g. 100ms")

	f.DurationVar(&config.Latencies.InterTokenLatencyStdDev, "inter-token-latency-std-dev", config.Latencies.InterTokenLatencyStdDev, "Standard deviation for time between generated tokens, e.g. 100ms")
	f.DurationVar(&config.Latencies.TimeToFirstTokenStdDev, "time-to-first-token-std-dev", config.Latencies.TimeToFirstTokenStdDev, "Standard deviation for time before the first token will be returned, e.g. 100ms")

	// The simulated KV-cache transfer of a disaggregated P/D deployment. Core
	// rather than engine-owned: the durations live in LatenciesConfig and the
	// latency model in pkg/simulator applies them whichever engine is running.
	f.DurationVar(&config.Latencies.KVCacheTransferLatency, "kv-cache-transfer-latency", config.Latencies.KVCacheTransferLatency, "Time for KV-cache transfer from a remote instance, e.g. 100ms")
	f.DurationVar(&config.Latencies.KVCacheTransferLatencyStdDev, "kv-cache-transfer-latency-std-dev", config.Latencies.KVCacheTransferLatencyStdDev, "Standard deviation for time for KV-cache transfer from a remote instance, e.g. 100ms")
	f.DurationVar(&config.Latencies.KVCacheTransferTimePerToken, "kv-cache-transfer-time-per-token", config.Latencies.KVCacheTransferTimePerToken, "Time for KV-cache transfer per token from a remote instance, e.g. 100ms")
	f.DurationVar(&config.Latencies.KVCacheTransferTimeStdDev, "kv-cache-transfer-time-std-dev", config.Latencies.KVCacheTransferTimeStdDev, "Standard deviation for time for KV-cache transfer per token from a remote instance, e.g. 100ms")
	f.Int64Var(&config.Seed, "seed", config.Seed, "Random seed for operations (if not set, current Unix time in nanoseconds is used)")
	f.Float64Var(&config.Latencies.TimeFactorUnderLoad, "time-factor-under-load", config.Latencies.TimeFactorUnderLoad, "Time factor under load (must be >= 1.0)")

	f.IntVar(&config.ToolCalls.MaxToolCallIntegerParam, "max-tool-call-integer-param", config.ToolCalls.MaxToolCallIntegerParam, "Maximum possible value of integer parameters in a tool call")
	f.IntVar(&config.ToolCalls.MinToolCallIntegerParam, "min-tool-call-integer-param", config.ToolCalls.MinToolCallIntegerParam, "Minimum possible value of integer parameters in a tool call")
	f.Float64Var(&config.ToolCalls.MaxToolCallNumberParam, "max-tool-call-number-param", config.ToolCalls.MaxToolCallNumberParam, "Maximum possible value of number (float) parameters in a tool call")
	f.Float64Var(&config.ToolCalls.MinToolCallNumberParam, "min-tool-call-number-param", config.ToolCalls.MinToolCallNumberParam, "Minimum possible value of number (float) parameters in a tool call")
	f.IntVar(&config.ToolCalls.MaxToolCallArrayParamLength, "max-tool-call-array-param-length", config.ToolCalls.MaxToolCallArrayParamLength, "Maximum possible length of array parameters in a tool call")
	f.IntVar(&config.ToolCalls.MinToolCallArrayParamLength, "min-tool-call-array-param-length", config.ToolCalls.MinToolCallArrayParamLength, "Minimum possible length of array parameters in a tool call")
	f.IntVar(&config.ToolCalls.ToolCallNotRequiredParamProbability, "tool-call-not-required-param-probability", config.ToolCalls.ToolCallNotRequiredParamProbability, "Probability to add a parameter, that is not required, in a tool call")
	f.IntVar(&config.ToolCalls.ObjectToolCallNotRequiredParamProbability, "object-tool-call-not-required-field-probability", config.ToolCalls.ObjectToolCallNotRequiredParamProbability, "Probability to add a field, that is not required, in an object in a tool call")
	f.IntVar(&config.ToolCalls.ToolCallExtraCallProbability, "tool-call-extra-call-probability", config.ToolCalls.ToolCallExtraCallProbability, "Probability (0-100) to make one additional tool call beyond the minimum; rolls repeat until a roll fails or all tools are called")

	f.IntVar(&config.DPSize, "data-parallel-size", config.DPSize, "Number of ranks to run")
	f.IntVar(&config.Rank, "data-parallel-rank", config.Rank, "The rank when running each rank in a process. If set, data-parallel-size is ignored")

	f.StringVar(&config.Dataset.DatasetPath, "dataset-path", config.Dataset.DatasetPath, "Local path to the sqlite db file for response generation from a dataset")
	f.StringVar(&config.Dataset.DatasetURL, "dataset-url", config.Dataset.DatasetURL, "URL to download the sqlite db file for response generation from a dataset")
	AddToggle(f, &config.Dataset.DatasetInMemory,
		"dataset-in-memory", "Load the entire dataset into memory for faster access", "Read the dataset from disk on demand")
	f.StringVar(&config.Dataset.DatasetTableName, "dataset-table-name", config.Dataset.DatasetTableName, "Table name for custom dataset, default is 'llmd'")

	f.StringVar(&config.RenderURL, "render-url", config.RenderURL, "URL of the tokenizer render service; when unset the simulated tokenizer is used")
	f.DurationVar(&config.RenderTimeout, "render-timeout", config.RenderTimeout, "Timeout for tokenizer render requests (e.g. 30s)")
	f.DurationVar(&config.MMRenderTimeout, "mm-render-timeout", config.MMRenderTimeout, "Timeout for multi-modal tokenizer render requests (e.g. 60s)")
	AddToggle(f, &config.ForceDummyTokenizer,
		"force-dummy-tokenizer", "(deprecated) Force the use of dummy tokenizer even if a real model name is provided; omit --render-url instead", "Use the tokenizer the model name implies")

	f.DurationVar(&config.StartupDuration, "startup-duration", config.StartupDuration,
		"Duration to return 503 on /health/ready to simulate GPU loading (e.g. 30s). Default is 0 (immediately ready)")

	AddToggle(f, &config.EnableRequestIDHeaders,
		"enable-request-id-headers", "Enable including X-Request-Id header in responses", "Omit the X-Request-Id header from responses")
	AddToggle(f, &config.LogHTTP,
		"log-http", "Log full HTTP request and response (method, URI, headers, bodies when buffered, status); streamed bodies are not logged", "Do not log full HTTP requests and responses")
	AddToggle(f, &config.ToolCalls.SkipToolValidation,
		"skip-tool-validation", "Skip the built-in validation of incoming tool schemas, matching real vLLM which forwards them to the model verbatim", "Validate incoming tool schemas")

	f.IntVar(&config.FailureInjectionRate, "failure-injection-rate", config.FailureInjectionRate, "Probability (0-100) of injecting failures")
	failureTypes := GetParamValueFromArgs("failure-types")
	var dummyFailureTypes multiString
	failureTypesDescription := fmt.Sprintf("List of specific failure types to inject (%s, %s, %s, %s, %s, %s)",
		FailureTypeRateLimit, FailureTypeInvalidAPIKey, FailureTypeContextLength, FailureTypeServerError, FailureTypeInvalidRequest,
		FailureTypeModelNotFound)
	f.Var(&dummyFailureTypes, "failure-types", failureTypesDescription)
	f.Lookup("failure-types").NoOptDefVal = dummy
	f.Lookup("failure-types").DefValue = ""

	f.StringVar(&config.SSL.SSLCertFile, "ssl-certfile", config.SSL.SSLCertFile, "Path to SSL certificate file for HTTPS (optional)")
	f.StringVar(&config.SSL.SSLKeyFile, "ssl-keyfile", config.SSL.SSLKeyFile, "Path to SSL private key file for HTTPS (optional)")
	AddToggle(f, &config.SSL.SelfSignedCerts,
		"self-signed-certs", "Enable automatic generation of self-signed certificates for HTTPS", "Do not generate self-signed certificates")

	f.StringVar(&config.LatencyCalculator, "latency-calculator", config.LatencyCalculator,
		`Name of the latency calculator to be used in the response generation (optional). The default calculation is based on the current load of the simulator and on
		the configured latency parameters, e.g., time-to-first-token and prefill-time-per-token`)

	f.IntVar(&config.DefaultEmbeddingDimensions, "default-embedding-dimensions", config.DefaultEmbeddingDimensions,
		"Default size of embedding vectors when the request does not specify dimensions (used by /v1/embeddings)")

	AddToggle(f, &config.Omni,
		"omni", "Enable omni mode: emit an image chunk when X-Send-Image header is present", "Disable omni mode")
	f.IntVar(&config.ImageEmissionRate, "image-emission-rate", config.ImageEmissionRate, "Probability (0-100) of emitting a synthetic image chunk per chat completion request in omni mode")
	f.DurationVar(&config.Latencies.TimeToGenerateImage, "time-to-generate-image", config.Latencies.TimeToGenerateImage, "Simulated time to generate an image in omni mode, e.g. 500ms")
	f.DurationVar(&config.Latencies.TimeToGenerateImageStdDev, "time-to-generate-image-std-dev", config.Latencies.TimeToGenerateImageStdDev, "Standard deviation for time to generate an image in omni mode, e.g. 50ms")

	// These values were manually parsed above in GetParamValueFromArgs, we leave this in order to get these flags in --help
	var dummyString string
	f.StringVar(&dummyString, "config", "", "The path to a yaml configuration file. The command line values overwrite the configuration file values")
	f.StringVar(&dummyString, "engine", "", "The inference engine to simulate: 'vllm' or 'sglang' (SGLang support is experimental)")
	var dummyMultiString multiString
	f.Var(&dummyMultiString, "served-model-name", "Model names exposed by the API (a list of space-separated strings)")
	// In order to allow empty arguments, we set a dummy NoOptDefVal for these flags
	f.Lookup("served-model-name").NoOptDefVal = dummy
	f.Lookup("served-model-name").DefValue = ""

	if err := eng.BindFlags(f, config, rawYAML); err != nil {
		return nil, err
	}
	// BindFlags has consumed the engine's own groups, so whatever is left in the
	// tree is a key no one claimed.
	if err := rejectUnknownYAMLKeys(rawYAML, config.EngineName); err != nil {
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

	if err := rejectSeparateBoolValue(f, os.Args[1:]); err != nil {
		return nil, err
	}
	if err := rejectUnexpectedArgs(f, os.Args[1:]); err != nil {
		return nil, err
	}

	// Set the values for Pod Name and Pod Namespace
	config.PodName = os.Getenv(PodNameEnv)
	config.PodNameSpace = os.Getenv(PodNsEnv)

	// Precedence for model: command-line flag > this env var > YAML > defaults.
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

	eng.ApplyEnv(config, f.Changed)

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
