package extproc

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Codex CLI sends the root id as session-id for both parent and review child.
// The root also sends thread-id; the child has its own thread-id. These cases
// cover tool continuations, a resumed user turn and isolation within one session.
func TestCodexCLIIdentityKeepsParentAndReviewChildModelsSeparate(t *testing.T) {
	const agentHeader = "thread-id"
	for _, tc := range []struct {
		name          string
		agentHeader   string
		wantSubagent  string
		wantAgentPart bool
	}{
		{name: "without_agent_header", wantSubagent: "frontier"},
		{name: "with_agent_header", agentHeader: agentHeader, wantSubagent: "cheap", wantAgentPart: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			registry := selection.NewRegistry()
			cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeSession)
			cfg.RouterLearning.Protection.Identity.Headers.Session = stringPtr("session-id")
			if tc.agentHeader != "" {
				cfg.RouterLearning.Protection.Identity.Headers.Agent = stringPtr(tc.agentHeader)
			}
			router := &OpenAIRouter{Config: cfg, ModelSelector: registry}
			both := []config.ModelRef{{Model: "cheap"}, {Model: "frontier"}}
			turn := 0
			selectRequest := func(agent, proposal string, tool bool) (string, *RequestContext) {
				t.Helper()
				if !tool {
					turn++
				}
				ctx := routerLearningRequestContext("run-1", "")
				ctx.Headers = map[string]string{"session-id": "run-1", "thread-id": agent}
				ctx.Headers["x-codex-turn-metadata"] = fmt.Sprintf(`{"session_id":"run-1","thread_id":%q,"turn_id":"turn-%d"}`, agent, turn)
				if agent != "run-1" {
					ctx.Headers["x-openai-subagent"] = "review"
					ctx.Headers["x-codex-parent-thread-id"] = "run-1"
				}
				ctx.VSRSelectedDecision = &config.Decision{Name: "choice"}
				ctx.VSRConversationFacts = classification.ConversationFacts{LastMessageToolResult: tool}
				registry.Register(selection.MethodStatic, selectionResultSelector{result: &selection.SelectionResult{
					SelectedModel: proposal, Score: 1, Confidence: 1, Method: selection.MethodStatic,
					AllScores: map[string]float64{proposal: 1},
				}})
				selCtx := router.buildSelectionContext(both, "choice", "", nil, "", nil, ctx)
				selected, _, err := router.selectModelFromCandidates(selCtx, nil, ctx)
				if err != nil || selected == nil {
					t.Fatalf("selection failed for agent=%q: selected=%+v err=%v", agent, selected, err)
				}
				if err := commitAgenticSessionDecision(ctx); err != nil {
					t.Fatal(err)
				}
				t.Logf("thread=%s proposal=%s selected=%s action=%s", agent, proposal, selected.Model, ctx.VSRLearningPolicy.String("action"))
				return selected.Model, ctx
			}

			parent, parentCtx := selectRequest("run-1", "frontier", false)
			if parent != "frontier" {
				t.Fatalf("main agent did not establish frontier: %q", parent)
			}
			if model, _ := selectRequest("run-1", "cheap", true); model != "frontier" {
				t.Fatalf("parent tool continuation changed pin: %q", model)
			}
			model, ctx := selectRequest("review-child", "cheap", false)
			if separate := parentCtx.VSRLearningSessionID != ctx.VSRLearningSessionID; separate != tc.wantAgentPart {
				t.Fatalf("parent/child key separation=%v, want %v", separate, tc.wantAgentPart)
			}
			if model != tc.wantSubagent {
				t.Fatalf("subagent first call got %q, want %q", model, tc.wantSubagent)
			}
			if part, ok := ctx.VSRLearningPolicy.ToMap()["identity"].(map[string]interface{})["agent"]; ok != tc.wantAgentPart {
				t.Fatalf("agent identity diagnostics present=%v, want %v: %#v", ok, tc.wantAgentPart, part)
			}
			if model, _ := selectRequest("review-child", "frontier", true); model != tc.wantSubagent {
				t.Errorf("subagent tool continuation got %q, want it held on %q", model, tc.wantSubagent)
			}
			if model, _ := selectRequest("run-1", "cheap", false); model != "frontier" {
				t.Errorf("resumed parent got %q after its subagent, want frontier", model)
			}
		})
	}
}
