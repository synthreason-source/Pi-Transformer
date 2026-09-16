#include <Arduino.h>
#include <cmath>

// Static Configuration Limits (No Heap Allocation)
#define MAX_TRANSITIONS 256
#define MAX_TOKEN_LEN 24

// Mathematical Hyperparameters
const float CURVE_K = 18.0f;
const float CURVE_MIDPOINT = 0.5f;

// Static Data Structures
struct Transition {
    char prev[MAX_TOKEN_LEN];
    char next[MAX_TOKEN_LEN];
    int count;
    float base_prob;
};

Transition transitions[MAX_TRANSITIONS];
int transition_count = 0;

enum SystemState { STATE_IDLE, STATE_RECEIVING_DATA };
SystemState current_state = STATE_IDLE;

// Zero-Heap Tokenizer using Fixed C-Buffers
int tokenizeLine(const char* line, char tokens[][MAX_TOKEN_LEN], int max_tokens) {
    int token_idx = 0;
    int char_idx = 0;
    
    for (int i = 0; line[i] != '\0' && token_idx < max_tokens; i++) {
        char c = line[i];
        if (c == ' ' || c == '\r' || c == '\n' || c == '\t') {
            if (char_idx > 0) {
                tokens[token_idx][char_idx] = '\0';
                token_idx++;
                char_idx = 0;
            }
        } else {
            if (char_idx < MAX_TOKEN_LEN - 1) {
                tokens[token_idx][char_idx++] = c;
            }
        }
    }
    if (char_idx > 0 && token_idx < max_tokens) {
        tokens[token_idx][char_idx] = '\0';
        token_idx++;
    }
    return token_idx;
}

// Ingest text line safely into static memory
void ingestTextLine(const char* line) {
    char tokens[16][MAX_TOKEN_LEN];
    int num_tokens = tokenizeLine(line, tokens, 16);
    if (num_tokens < 2) return;

    for (int i = 0; i < num_tokens - 1; ++i) {
        bool found = false;
        for (int j = 0; j < transition_count; j++) {
            if (strcmp(transitions[j].prev, tokens[i]) == 0 && 
                strcmp(transitions[j].next, tokens[i+1]) == 0) {
                transitions[j].count++;
                found = true;
                break;
            }
        }
        if (!found) {
            if (transition_count < MAX_TRANSITIONS) {
                strncpy(transitions[transition_count].prev, tokens[i], MAX_TOKEN_LEN - 1);
                transitions[transition_count].prev[MAX_TOKEN_LEN - 1] = '\0';
                
                strncpy(transitions[transition_count].next, tokens[i+1], MAX_TOKEN_LEN - 1);
                transitions[transition_count].next[MAX_TOKEN_LEN - 1] = '\0';
                
                transitions[transition_count].count = 1;
                transitions[transition_count].base_prob = 0.0f;
                transition_count++;
            } else {
                Serial.println("[Warning] Transition table capacity reached.");
                break;
            }
        }
    }
}

// Finalize probabilities using static indexing
void finalizeDataset() {
    for (int i = 0; i < transition_count; i++) {
        int totalContextCount = 0;
        for (int j = 0; j < transition_count; j++) {
            if (strcmp(transitions[j].prev, transitions[i].prev) == 0) {
                totalContextCount += transitions[j].count;
            }
        }
        if (totalContextCount > 0) {
            transitions[i].base_prob = (float)transitions[i].count / (float)totalContextCount;
        }
    }
    Serial.printf("[Dataset] Finalized. Statically loaded %d transitions.\n", transition_count);
}

// Math 1: Compute Semantic Similarity via Transition Overlaps
float computeSemanticSimilarity(const char* wordA, const char* wordB) {
    float sharedOverlap = 0.0f;
    float totalA = 0.0f;
    float totalB = 0.0f;

    for (int i = 0; i < transition_count; i++) {
        if (strcmp(transitions[i].prev, wordA) == 0) totalA += transitions[i].base_prob;
        if (strcmp(transitions[i].prev, wordB) == 0) totalB += transitions[i].base_prob;
        if (strcmp(transitions[i].next, wordA) == 0 && strcmp(transitions[i].next, wordB) == 0) {
            sharedOverlap += 0.5f;
        }
    }

    if (totalA <= 0.0f || totalB <= 0.0f) return 0.1f;
    return fminf(1.0f, sharedOverlap / sqrtf(totalA * totalB) + 0.2f);
}

// Math 2: Sigmoid Curve Transformation
float sigmoidCurve(float value, float k, float midpoint) {
    return 1.0f / (1.0f + expf(-k * (value - midpoint)));
}

// Real-Time Static Math Inference Loop
void processPromptWithMath(const char* inputPrompt) {
    if (transition_count == 0) {
        Serial.println("[Error] Dataset is empty! Send 'UPLOAD_START' first.");
        return;
    }

    char tokens[16][MAX_TOKEN_LEN];
    int num_tokens = tokenizeLine(inputPrompt, tokens, 16);
    if (num_tokens == 0) return;

    char currentContext[MAX_TOKEN_LEN];
    strncpy(currentContext, tokens[num_tokens - 1], MAX_TOKEN_LEN - 1);
    currentContext[MAX_TOKEN_LEN - 1] = '\0';

    char outputBuffer[256];
    snprintf(outputBuffer, sizeof(outputBuffer), "%s", inputPrompt);

    Serial.println("\n--- Static-Memory Mathematical Inference ---");
    Serial.printf("Context: %s\n", currentContext);

    for (int step = 0; step < 5; step++) {
        char bestNext[MAX_TOKEN_LEN] = "";
        float maxScore = -1e9f;

        for (int i = 0; i < transition_count; i++) {
            if (strcmp(transitions[i].prev, currentContext) == 0) {
                float similarity = computeSemanticSimilarity(currentContext, transitions[i].next);
                float curveWeight = sigmoidCurve(transitions[i].base_prob, CURVE_K, CURVE_MIDPOINT);
                
                float score = logf(fmaxf(transitions[i].base_prob, 1e-12f)) + (curveWeight * 0.5f * similarity);
                float thermalNoise = ((float)esp_random() / (float)UINT32_MAX) * 0.05f;
                score += thermalNoise;

                if (score > maxScore) {
                    maxScore = score;
                    strncpy(bestNext, transitions[i].next, MAX_TOKEN_LEN - 1);
                    bestNext[MAX_TOKEN_LEN - 1] = '\0';
                }
            }
        }

        if (bestNext[0] == '\0') break;

        strcat(outputBuffer, " ");
        strcat(outputBuffer, bestNext);
        strncpy(currentContext, bestNext, MAX_TOKEN_LEN - 1);
        currentContext[MAX_TOKEN_LEN - 1] = '\0';
    }

    Serial.printf("Output: %s\n", outputBuffer);
    Serial.println("---------------------------------------------\n");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("[ESP32-C3] Zero-Heap Static Math Engine Ready.");
}

void loop() {
    if (Serial.available() > 0) {
        String inputStr = Serial.readStringUntil('\n');
        inputStr.trim();
        if (inputStr.length() == 0) return;

        if (inputStr == "UPLOAD_START") {
            current_state = STATE_RECEIVING_DATA;
            transition_count = 0;
            Serial.println("[System] Ready for text lines. Send 'UPLOAD_END' when finished.");
            return;
        }

        if (inputStr == "UPLOAD_END") {
            current_state = STATE_IDLE;
            finalizeDataset();
            return;
        }

        if (current_state == STATE_RECEIVING_DATA) {
            ingestTextLine(inputStr.c_str());
        } else {
            processPromptWithMath(inputStr.c_str());
        }
    }
}
