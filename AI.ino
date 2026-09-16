#include <Arduino.h>
#include <cmath>

// Static Configuration Limits (Zero Heap Allocation)
#define MAX_WORDS 64
#define VECTOR_DIM 8
#define MAX_TRANSITIONS 256
#define MAX_TOKEN_LEN 24

// Mathematical Hyperparameters
const float CURVE_K = 18.0f;
const float CURVE_MIDPOINT = 0.5f;

// Static Data Structures
struct WordEmbedding {
    char word[MAX_TOKEN_LEN];
    float vector[VECTOR_DIM];
};

struct Transition {
    char prev[MAX_TOKEN_LEN];
    char next[MAX_TOKEN_LEN];
    int count;
    float base_prob;
};

WordEmbedding vocab[MAX_WORDS];
int vocab_count = 0;

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

// Generate or retrieve a deterministic embedding vector for a word (Zero Heap)
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
        
        // Generate pseudo-embedding feature vector via character hashing/trigonometry
        for (int d = 0; d < VECTOR_DIM; d++) {
            float val = 0.0f;
            for (int c = 0; word[c] != '\0'; c++) {
                val += sinf((float)(c + 1) * (d + 1) * word[c]);
            }
            vocab[vocab_count].vector[d] = val / (float)(strlen(word) + 1);
        }

        for (int d = 0; d < VECTOR_DIM; d++) out_vec[d] = vocab[vocab_count].vector[d];
        vocab_count++;
    } else {
        for (int d = 0; d < VECTOR_DIM; d++) out_vec[d] = 0.1f;
    }
}

// Math 1: True Vector Cosine Similarity
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

// Ingest text line into static transitions and build embeddings
void ingestTextLine(const char* line) {
    char tokens[16][MAX_TOKEN_LEN];
    int num_tokens = tokenizeLine(line, tokens, 16);
    if (num_tokens < 2) return;

    for (int i = 0; i < num_tokens - 1; ++i) {
        // Ensure embeddings exist for vector math
        float dummy[VECTOR_DIM];
        getOrCreateEmbedding(tokens[i], dummy);
        getOrCreateEmbedding(tokens[i+1], dummy);

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
            }
        }
    }
}

// Finalize dataset probabilities
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
    Serial.printf("[Dataset] Finalized. %d transitions, %d vocabulary tokens loaded.\n", 
                  transition_count, vocab_count);
}

// Real-Time Math Inference Loop (Cosine Similarity + Sigmoid + Noise)
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

    Serial.println("\n--- Vector Cosine & Sigmoid Inference ---");
    Serial.printf("Context: %s\n", currentContext);

    for (int step = 0; step < 5; step++) {
        char bestNext[MAX_TOKEN_LEN] = "";
        float maxScore = -1e9f;

        for (int i = 0; i < transition_count; i++) {
            if (strcmp(transitions[i].prev, currentContext) == 0) {
                // Math Execution: Cosine Similarity & Sigmoid Scaling
                float similarity = computeCosineSimilarity(currentContext, transitions[i].next);
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
    Serial.println------------------------------------------------\n");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("[ESP32-C3] Zero-Heap Vector Math Engine Ready.");
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
