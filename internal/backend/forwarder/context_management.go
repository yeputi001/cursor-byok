package forwarder

import (
	"encoding/json"
	"strings"

	modeladapter "cursor/internal/backend/agent/model"
)

// Context window and trimming constants mirror Claude Code's context.ts and
// autoCompact.ts (see 04f-context-management.md).
const (
	contextAutoCompactBufferTokens   = 13_000
	contextSummaryReserveTokens      = 20_000
	contextMaxConsecutiveCompactFail = 3
	contextMaxPTLRetries             = 3
	contextMaxTooLongRetries           = 2

	contextMaxToolMessageBytes      = 24 * 1024
	contextMaxWorkspaceContextBytes = 48 * 1024
	contextMaxCompiledMessageBytes  = 32 * 1024
	contextMaxToolsTotalBytes       = 96 * 1024
	contextTrimMessageThreshold     = 35
	contextEarlyTrimFraction        = 0.50
	contextAutoCompactUsageFraction = 0.75
	contextMinOutputReserveTokens   = 16_000
	contextMaxInLoopTailMessages    = 28

	// Estimates use rune/4; real Claude tokenizers often count code/JSON higher.
	contextEstimateSafetyNumerator   = 7
	contextEstimateSafetyDenominator = 5
)

// effectiveCompiledContextWindow returns the token budget available to the
// agent after reserving headroom for a potential compaction summary request.
func effectiveCompiledContextWindow(conversation *ConversationFile) int64 {
	total := compactionContextWindowSize(conversation)
	reserved := int64(contextSummaryReserveTokens)
	if reserved >= total {
		return total / 2
	}
	return total - reserved
}

// resolveGatedPromptTokens returns the token count used for trim/compact
// decisions. It prefers the higher of: safety-adjusted estimate, last known
// provider usage, and pending auto-compaction telemetry.
func resolveGatedPromptTokens(compiled CompiledConversation, conversation *ConversationFile) int64 {
	estimated := applyContextEstimateSafety(estimateCompiledPromptTokens(compiled))
	if conversation == nil {
		return estimated
	}
	gated := maxPositiveInt64(
		estimated,
		int64(conversation.TokenDetailsUsedTokens),
		conversation.AutoCompactionPromptTokens,
	)
	if prefix := conversation.LatestRequestPrefix; prefix != nil {
		if strings.TrimSpace(conversation.CurrentRequestID) != "" &&
			strings.TrimSpace(prefix.RequestID) == strings.TrimSpace(conversation.CurrentRequestID) &&
			prefix.PromptTokensTotal > gated {
			gated = prefix.PromptTokensTotal
		}
	}
	return gated
}

func applyContextEstimateSafety(tokens int64) int64 {
	if tokens <= 0 {
		return 0
	}
	return tokens * int64(contextEstimateSafetyNumerator) / int64(contextEstimateSafetyDenominator)
}

// providerInputBudgetTokens is the hard input ceiling before a provider call.
func providerInputBudgetTokens(conversation *ConversationFile) int64 {
	effective := effectiveCompiledContextWindow(conversation)
	budget := effective - int64(contextAutoCompactBufferTokens)
	if budget < 8_000 {
		budget = 8_000
	}
	return budget
}

// trimBudgetTarget is the proactive token count we trim toward before the
// upstream provider rejects the request.
func trimBudgetTarget(effectiveWindow int64) int64 {
	if effectiveWindow <= 0 {
		return 80_000
	}
	target := int64(float64(effectiveWindow) * contextEarlyTrimFraction)
	if target < 8_000 {
		return 8_000
	}
	return target
}

// shouldTrimCompiledContext returns true when the live prompt should shrink
// proactively — earlier than auto-compact, and also on high message counts.
func shouldTrimCompiledContext(tokensUsed, effectiveWindow int64, messageCount int) bool {
	if messageCount >= contextTrimMessageThreshold {
		return true
	}
	if effectiveWindow <= 0 {
		return messageCount > 20 || tokensUsed > 80_000
	}
	early := trimBudgetTarget(effectiveWindow)
	return tokensUsed >= early || shouldAutoCompactByEstimate(tokensUsed, effectiveWindow, 0)
}

// autoCompactTriggerThreshold is when auto-compact should fire. Uses the lower
// of (effective - buffer) and 75% of effective so 143k on a 200k model triggers
// compact around ~135k instead of waiting until ~167k.
func autoCompactTriggerThreshold(effectiveWindow int64) int64 {
	if effectiveWindow <= 0 {
		return 0
	}
	byBuffer := effectiveWindow - int64(contextAutoCompactBufferTokens)
	byFraction := int64(float64(effectiveWindow) * contextAutoCompactUsageFraction)
	threshold := byBuffer
	if byFraction < threshold {
		threshold = byFraction
	}
	floor := effectiveWindow / 2
	if threshold < floor {
		threshold = floor
	}
	return threshold
}

// shouldAutoCompactByEstimate returns true when estimated usage crosses the
// buffer threshold inside the effective window. snipFreed adjusts the estimate
// when a cheap head-snip already dropped tokens this round.
func shouldAutoCompactByEstimate(tokensUsed, effectiveWindow, snipFreed int64) bool {
	if effectiveWindow <= 0 {
		return false
	}
	return tokensUsed-snipFreed >= autoCompactTriggerThreshold(effectiveWindow)
}

// shouldAutoCompactByOutputPressure triggers when remaining headroom cannot
// sustain a useful model response (explains output=8 with oversized input).
func shouldAutoCompactByOutputPressure(gated, effectiveWindow int64) bool {
	if effectiveWindow <= 0 || gated <= 0 {
		return false
	}
	return effectiveWindow-gated < int64(contextMinOutputReserveTokens)
}

// isContextTooLong reports upstream rejections that mean the prompt must shrink.
func isContextTooLong(err error) bool {
	if err == nil {
		return false
	}
	if isPromptTooLong(err) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "input is too long") ||
		strings.Contains(msg, "content_length_exceeds") ||
		strings.Contains(msg, "context length exceeded") ||
		strings.Contains(msg, "request too large") ||
		strings.Contains(msg, "maximum context") ||
		strings.Contains(msg, "token limit")
}

// isPromptTooLong reports compaction-summary and provider rejections that mean
// the summarization input itself must shrink.
func isPromptTooLong(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "prompt is too long") ||
		strings.Contains(msg, "prompt too long") ||
		strings.Contains(msg, "context length") ||
		strings.Contains(msg, "maximum context") ||
		strings.Contains(msg, "token limit") ||
		strings.Contains(msg, "413")
}

type contextTrimResult struct {
	Trimmed     bool
	TokensFreed int64
	SnipFreed   int64
}

// manageCompiledContextBeforeProvider applies cheap snip and emergency trim
// when the live message list approaches the effective window. Trimming only
// affects the ephemeral provider request — persisted history entries are not
// rewritten.
func manageCompiledContextBeforeProvider(compiled CompiledConversation, conversation *ConversationFile, aggressive bool) (CompiledConversation, contextTrimResult) {
	var result contextTrimResult
	if len(compiled.Messages) < 2 {
		return compiled, result
	}
	beforeGated := resolveGatedPromptTokens(compiled, conversation)
	budget := providerInputBudgetTokens(conversation)
	targetTokens := trimBudgetTarget(effectiveCompiledContextWindow(conversation))
	if aggressive {
		targetTokens = budget / 2
		if targetTokens < 4_000 {
			targetTokens = 4_000
		}
	}
	effective := effectiveCompiledContextWindow(conversation)
	needsTrim := aggressive ||
		beforeGated > budget ||
		shouldTrimCompiledContext(beforeGated, effective, len(compiled.Messages))
	if !needsTrim {
		return compiled, result
	}
	if targetTokens > budget {
		targetTokens = budget
	}
	compiled, freed := shrinkCompiledConversation(compiled, targetTokens)
	if freed > 0 {
		result.Trimmed = true
		result.TokensFreed = freed
	}
	afterGated := resolveGatedPromptTokens(compiled, conversation)
	if afterGated > budget {
		var extraFreed int64
		compiled, extraFreed = shrinkCompiledConversation(compiled, budget/2)
		if extraFreed > 0 {
			result.Trimmed = true
			result.TokensFreed += extraFreed
		}
	}
	return compiled, result
}

// enforceProviderInputBudget keeps trimming until gated usage fits the hard
// provider input ceiling or no further progress is possible.
func enforceProviderInputBudget(compiled CompiledConversation, conversation *ConversationFile) CompiledConversation {
	budget := providerInputBudgetTokens(conversation)
	for pass := 0; pass < 5; pass++ {
		gated := resolveGatedPromptTokens(compiled, conversation)
		if gated <= budget {
			return compiled
		}
		target := budget
		if pass > 0 {
			target = budget / int64(pass+1)
			if target < 4_000 {
				target = 4_000
			}
		}
		before := estimateCompiledPromptTokens(compiled)
		next, _ := shrinkCompiledConversation(compiled, target)
		after := estimateCompiledPromptTokens(next)
		if after >= before {
			break
		}
		compiled = next
	}
	return compiled
}

func shrinkCompiledConversation(compiled CompiledConversation, targetTokens int64) (CompiledConversation, int64) {
	if len(compiled.Messages) < 2 || targetTokens <= 0 {
		return compiled, 0
	}
	before := estimateCompiledPromptTokens(compiled)
	compiled.Tools = capCompiledToolDescriptors(compiled.Tools)
	messages := capCompiledMessageBodies(compiled.Messages)
	messages = truncateAllCompiledToolMessages(messages)
	messages = capCompiledWorkspaceMessages(messages)

	prefixLen := compiledPromptPrefixLen(compiled, messages)
	messages = emergencyTrimCompiledMessages(messages, prefixLen, targetTokens)
	compiled.Messages = messages

	after := estimateCompiledPromptTokens(compiled)
	freed := before - after
	if freed < 0 {
		freed = 0
	}
	return compiled, freed
}

func compiledPromptPrefixLen(compiled CompiledConversation, messages []modeladapter.Message) int {
	stable := compiled.StableMessageCount
	if stable < 0 {
		stable = 0
	}
	prefix := 1 + stable
	if prefix > len(messages) {
		prefix = len(messages)
	}
	if prefix < 1 {
		prefix = 1
	}
	return prefix
}

// cheapSnipOldCompiledMessages drops the oldest replayed history messages
// (after the static system prefix) until estimated tokens fall below target
// or no more history remains. Returns tokens freed (estimate).
func cheapSnipOldCompiledMessages(messages []modeladapter.Message, prefixLen int, targetTokens int64) ([]modeladapter.Message, int64) {
	if len(messages) < 2 {
		return messages, 0
	}
	before := estimateModelMessagesTokens(messages)
	if prefixLen < 2 {
		return trimCompiledInLoopTail(messages, 1, contextMaxInLoopTailMessages), before - estimateModelMessagesTokens(messages)
	}
	histStart := 1
	if histStart >= len(messages) {
		return messages, 0
	}
	histEnd := prefixLen
	if histEnd > len(messages) {
		histEnd = len(messages)
	}
	if histEnd <= histStart {
		return trimCompiledInLoopTail(messages, 1, contextMaxInLoopTailMessages), before - estimateModelMessagesTokens(messages)
	}
	out := append([]modeladapter.Message(nil), messages[:histStart]...)
	hist := append([]modeladapter.Message(nil), messages[histStart:histEnd]...)
	tail := append([]modeladapter.Message(nil), messages[histEnd:]...)

	for len(hist) > 0 && estimateModelMessagesTokens(append(append(out, hist...), tail...)) > targetTokens {
		drop := 1
		if len(hist) >= 2 {
			drop = 2
		}
		hist = hist[drop:]
	}
	result := append(out, hist...)
	result = append(result, tail...)
	after := estimateModelMessagesTokens(result)
	freed := before - after
	if freed < 0 {
		freed = 0
	}
	return result, freed
}

// emergencyTrimCompiledMessages aggressively shrinks the live prompt: cap tool
// payloads, shrink workspace blocks, snip old history, then drop in-loop tail
// rounds until under targetTokens.
func emergencyTrimCompiledMessages(messages []modeladapter.Message, prefixLen int, targetTokens int64) []modeladapter.Message {
	if len(messages) < 2 || targetTokens <= 0 {
		return messages
	}
	out := messages
	if estimateModelMessagesTokens(out) <= targetTokens {
		return out
	}
	snipped, _ := cheapSnipOldCompiledMessages(out, prefixLen, targetTokens)
	out = snipped

	for keep := contextMaxInLoopTailMessages; keep >= 2 && estimateModelMessagesTokens(out) > targetTokens; keep -= 4 {
		out = trimCompiledInLoopTail(out, maxInt(1, prefixLen), keep)
	}
	for estimateModelMessagesTokens(out) > targetTokens && len(out) > 2 {
		out = trimCompiledInLoopTail(out, 1, 2)
	}
	return out
}

func capCompiledMessageBodies(messages []modeladapter.Message) []modeladapter.Message {
	out := make([]modeladapter.Message, len(messages))
	for i, message := range messages {
		out[i] = message
		if strings.TrimSpace(message.Role) == "system" {
			continue
		}
		if text := strings.TrimSpace(message.Content); text != "" && len(text) > contextMaxCompiledMessageBytes {
			out[i].Content = truncateCompiledText(text, contextMaxCompiledMessageBytes, "message body")
		}
		if len(message.ToolCalls) == 0 {
			continue
		}
		toolCalls := make([]modeladapter.ToolCallDescriptor, len(message.ToolCalls))
		for j, toolCall := range message.ToolCalls {
			toolCalls[j] = toolCall
			if len(toolCall.Function.Arguments) > contextMaxCompiledMessageBytes {
				toolCalls[j].Function.Arguments = truncateCompiledText(
					toolCall.Function.Arguments,
					contextMaxCompiledMessageBytes,
					"tool arguments",
				)
			}
		}
		out[i].ToolCalls = toolCalls
	}
	return out
}

func capCompiledToolDescriptors(tools []json.RawMessage) []json.RawMessage {
	if len(tools) == 0 {
		return tools
	}
	total := 0
	out := make([]json.RawMessage, 0, len(tools))
	for _, tool := range tools {
		text := string(tool)
		if total+len(text) > contextMaxToolsTotalBytes {
			break
		}
		out = append(out, tool)
		total += len(text)
	}
	return out
}

func truncateCompiledText(text string, maxBytes int, label string) string {
	if maxBytes <= 0 || len(text) <= maxBytes {
		return text
	}
	marker := "\n…[" + strings.TrimSpace(label) + " truncated to fit model limit]"
	cut := maxBytes - len(marker)
	if cut < 256 {
		cut = maxBytes
		marker = ""
	}
	return text[:cut] + marker
}

func truncateCompiledToolPayload(msg modeladapter.Message, maxBytes int) modeladapter.Message {
	if strings.TrimSpace(msg.Role) != "tool" || maxBytes <= 0 {
		return msg
	}
	text := strings.TrimSpace(msg.Content)
	if len(text) <= maxBytes {
		return msg
	}
	out := msg
	out.Content = truncateCompiledText(text, maxBytes, "tool output")
	return out
}

func truncateAllCompiledToolMessages(messages []modeladapter.Message) []modeladapter.Message {
	out := make([]modeladapter.Message, len(messages))
	for i, m := range messages {
		out[i] = truncateCompiledToolPayload(m, contextMaxToolMessageBytes)
	}
	return out
}

func capCompiledWorkspaceMessages(messages []modeladapter.Message) []modeladapter.Message {
	if len(messages) == 0 {
		return messages
	}
	out := append([]modeladapter.Message(nil), messages...)
	for i := range out {
		if strings.TrimSpace(out[i].Role) != "user" {
			continue
		}
		text := strings.TrimSpace(out[i].Content)
		if text == "" {
			continue
		}
		if strings.Contains(text, "<user_info>") && len(text) > contextMaxWorkspaceContextBytes {
			out[i].Content = truncateCompiledText(text, contextMaxWorkspaceContextBytes, "workspace context")
		}
	}
	return out
}

func trimCompiledInLoopTail(messages []modeladapter.Message, prefixLen, maxKeep int) []modeladapter.Message {
	if maxKeep <= 0 || prefixLen <= 0 || len(messages) <= prefixLen {
		return messages
	}
	tail := messages[prefixLen:]
	if len(tail) <= maxKeep {
		return messages
	}
	drop := len(tail) - maxKeep
	out := append([]modeladapter.Message(nil), messages[:prefixLen]...)
	out = append(out, tail[drop:]...)
	return out
}

func stripImagesFromCompactionText(text string) string {
	for {
		start := strings.Index(text, "data:image/")
		if start < 0 {
			break
		}
		end := strings.Index(text[start:], ";base64,")
		if end < 0 {
			break
		}
		rest := text[start+end+8:]
		close := strings.IndexAny(rest, "\"'\n> ")
		if close < 0 {
			text = text[:start] + "[image]" + rest
		} else {
			text = text[:start] + "[image]" + rest[close:]
		}
	}
	return text
}

func peelCompactionSummaryInputForPTL(text string) (string, bool) {
	cut := len(text) / 5
	if cut < 512 {
		return text, false
	}
	return text[cut:], true
}

func autoCompactionCircuitOpen(conversation *ConversationFile) bool {
	if conversation == nil {
		return false
	}
	return conversation.AutoCompactionConsecutiveFailures >= contextMaxConsecutiveCompactFail
}

func recordAutoCompactionSuccess(conversation *ConversationFile) {
	if conversation == nil {
		return
	}
	conversation.AutoCompactionConsecutiveFailures = 0
}

func recordAutoCompactionFailure(conversation *ConversationFile) {
	if conversation == nil {
		return
	}
	conversation.AutoCompactionConsecutiveFailures++
}

func contextUsagePercent(tokensUsed, effectiveWindow int64) float64 {
	if effectiveWindow <= 0 {
		return 0
	}
	pct := float64(tokensUsed) / float64(effectiveWindow) * 100
	if pct > 100 {
		return 100
	}
	if pct < 0 {
		return 0
	}
	return pct
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
