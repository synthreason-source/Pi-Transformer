#include <Arduino.h>
#include <cmath>

// Expanded Static Configuration Limits (Optimized for ESP32-C3 SRAM)
#define MAX_WORDS 512
#define VECTOR_DIM 16
#define MAX_TRANSITIONS 2048
#define MAX_TOKEN_LEN 24
#define MAX_LINE_TOKENS 64

// Mathematical Hyperparameters
const float CURVE_K = 18.0f;
const float CURVE_MIDPOINT = 0.5f;

// Static Data Structures
struct WordEmbedding {
    char word[MAX_TOKEN_LEN];
    float vector[VECTOR_DIM];
};

struct TrigramTransition {
    char prev1[MAX_TOKEN_LEN];
    char prev2[MAX_TOKEN_LEN];
    char next[MAX_TOKEN_LEN];
    int count;
    float base_prob;
};

WordEmbedding vocab[MAX_WORDS];
int vocab_count = 0;

TrigramTransition transitions[MAX_TRANSITIONS];
int transition_count = 0;

enum SystemState { STATE_IDLE, STATE_RECEIVING_DATA };
SystemState current_state = STATE_IDLE;

// Expanded Zero-Heap Tokenizer (Supports up to MAX_LINE_TOKENS per read)
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

// Generate or retrieve a deterministic 16-D embedding vector for a word (Zero Heap)
void getOrCreateEmbedding(const char* word, float* out_vec) {
    for (int i = 0; i < vocab_count; i++) {
        if (strcmp(vocab[i].word, word) == 0) {
            for (int d = 0; d < VECTOR_DIM; d++) out_vec[d] = vocab[i].vector[d];
            return;
        }
    }

    if (vocab_count < MAX_WORDS) {
        strncpy(vocab[vocab_count].word, word, MAX_TOKEN_LEN - 1);
        vocab[vocab_count].word[MAX_TOKEN_LEN - 1] = '\0';
        
        // Generate expanded 16-D pseudo-embedding via trigonometric character hashing
        for (int d = 0; d < VECTOR_DIM; d++) {
            float val = 0.0f;
            for (int c = 0; word[c] != '\0'; c++) {
                val += sinf((float)(c + 1) * (d + 1) * word[c]) * cosf((float)c / (d + 1.0f));
            }
            vocab[vocab_count].vector[d] = val / (float)(strlen(word) + 1);
        }

        for (int d = 0; d < VECTOR_DIM; d++) out_vec[d] = vocab[vocab_count].vector[d];
        vocab_count++;
    } else {
        for (int d = 0; d < VECTOR_DIM; d++) out_vec[d] = 0.1f;
    }
}

// Math 1: 16-D Vector Cosine Similarity
float computeCosineSimilarity(const char* wordA, const char* wordB) {
    float vecA[VECTOR_DIM];
    float vecB[VECTOR_DIM];
    getOrCreateEmbedding(wordA, vecA);
    getOrCreateEmbedding(wordB, vecB);

    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (int d = 0; d < VECTOR_DIM; d++) {
        dot += vecA[d] * vecB[d];
        normA += vecA[d] * vecA[d];
        normB += vecB[d] * vecB[d];
    }

    if (normA <= 0.0f || normB <= 0.0f) return 0.0f;
    return dot / (sqrtf(normA) * sqrtf(normB));
}

// Math 2: Sigmoid Curve Transformation
float sigmoidCurve(float value, float k, float midpoint) {
    return 1.0f / (1.0f + expf(-k * (value - midpoint)));
}

// Ingest text line into static trigram transitions and build embeddings
void ingestTextLine(const char* line) {
    char tokens[MAX_LINE_TOKENS][MAX_TOKEN_LEN];
    int num_tokens = tokenizeLine(line, tokens, MAX_LINE_TOKENS);
    if (num_tokens < 3) return;

    for (int i = 0; i < num_tokens - 2; ++i) {
        float dummy[VECTOR_DIM];
        getOrCreateEmbedding(tokens[i], dummy);
        getOrCreateEmbedding(tokens[i+1], dummy);
        getOrCreateEmbedding(tokens[i+2], dummy);

        bool found = false;
        for (int j = 0; j < transition_count; j++) {
            if (strcmp(transitions[j].prev1, tokens[i]) == 0 && 
                strcmp(transitions[j].prev2, tokens[i+1]) == 0 &&
                strcmp(transitions[j].next, tokens[i+2]) == 0) {
                transitions[j].count++;
                found = true;
                break;
            }
        }
        if (!found) {
            if (transition_count < MAX_TRANSITIONS) {
                strncpy(transitions[transition_count].prev1, tokens[i], MAX_TOKEN_LEN - 1);
                transitions[transition_count].prev1[MAX_TOKEN_LEN - 1] = '\0';

                strncpy(transitions[transition_count].prev2, tokens[i+1], MAX_TOKEN_LEN - 1);
                transitions[transition_count].prev2[MAX_TOKEN_LEN - 1] = '\0';
                
                strncpy(transitions[transition_count].next, tokens[i+2], MAX_TOKEN_LEN - 1);
                transitions[transition_count].next[MAX_TOKEN_LEN - 1] = '\0';
                
                transitions[transition_count].count = 1;
                transitions[transition_count].base_prob = 0.0f;
                transition_count++;
            } else {
                static bool warned = false;
                if (!warned) {
                    Serial.println("[Warning] MAX_TRANSITIONS capacity reached! Increase limit if needed.");
                    warned = true;
                }
                break;
            }
        }
    }
}

// Finalize dataset probabilities
void finalizeDataset() {
    for (int i = 0; i < transition_count; i++) {
        int totalContextCount = 0;
        for (int j = 0; j < transition_count; j++) {
            if (strcmp(transitions[j].prev1, transitions[i].prev1) == 0 &&
                strcmp(transitions[j].prev2, transitions[i].prev2) == 0) {
                totalContextCount += transitions[j].count;
            }
        }
        if (totalContextCount > 0) {
            transitions[i].base_prob = (float)transitions[i].count / (float)totalContextCount;
        }
    }
    Serial.printf("[Dataset] Trigram Engine Finalized. %d transitions, %d vocabulary tokens loaded.\n", 
                  transition_count, vocab_count);
}

// Real-Time Trigram Math Inference Loop
void processPromptWithMath(const char* inputPrompt) {
    if (transition_count == 0) {
        Serial.println("[Error] Dataset is empty! Send 'UPLOAD_START' first.");
        return;
    }

    char tokens[MAX_LINE_TOKENS][MAX_TOKEN_LEN];
    int num_tokens = tokenizeLine(inputPrompt, tokens, MAX_LINE_TOKENS);
    if (num_tokens < 2) {
        Serial.println("[Error] Provide at least 2 context words for trigram inference.");
        return;
    }

    char ctx1[MAX_TOKEN_LEN];
    char ctx2[MAX_TOKEN_LEN];
    strncpy(ctx1, tokens[num_tokens - 2], MAX_TOKEN_LEN - 1);
    strncpy(ctx2, tokens[num_tokens - 1], MAX_TOKEN_LEN - 1);
    ctx1[MAX_TOKEN_LEN - 1] = '\0';
    ctx2[MAX_TOKEN_LEN - 1] = '\0';

    char outputBuffer[256];
    snprintf(outputBuffer, sizeof(outputBuffer), "%s", inputPrompt);

    Serial.println("\n--- 16-D Trigram Cosine & Sigmoid Inference ---");
    Serial.printf("Context Window: [%s, %s]\n", ctx1, ctx2);

    for (int step = 0; step < 6; step++) {
        char bestNext[MAX_TOKEN_LEN] = "";
        float maxScore = -1e9f;

        for (int i = 0; i < transition_count; i++) {
            if (strcmp(transitions[i].prev1, ctx1) == 0 && strcmp(transitions[i].prev2, ctx2) == 0) {
                float similarity = computeCosineSimilarity(ctx2, transitions[i].next);
                float curveWeight = sigmoidCurve(transitions[i].base_prob, CURVE_K, CURVE_MIDPOINT);
                
                float score = logf(fmaxf(transitions[i].base_prob, 1e-12f)) + (curveWeight * 0.6f * similarity);
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
        
        strncpy(ctx1, ctx2, MAX_TOKEN_LEN - 1);
        strncpy(ctx2, bestNext, MAX_TOKEN_LEN - 1);
    }

    Serial.printf("Output: %s\n", outputBuffer);
    Serial.println("-----------------------------------------------\n");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("[ESP32-C3] High-Capacity Zero-Heap Trigram Engine Ready.");
}

void loop() {
    if (Serial.available() > 0) {
        String inputStr = Serial.readStringUntil('\n');
        inputStr.trim();
        if (inputStr.length() == 0) return;

        if (inputStr == "UPLOAD_START") {
            current_state = STATE_RECEIVING_DATA;
            transition_count = 0;
            vocab_count = 0;
            Serial.println("[System] Ready for large dataset upload. Send 'UPLOAD_END' when finished.");
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
