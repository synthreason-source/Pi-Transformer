#include <Arduino.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>

// Mathematical Hyperparameters
const float CURVE_K = 18.0f;
const float CURVE_MIDPOINT = 0.5f;

// Dynamic Data Structures
struct Transition {
    std::string prev;
    std::string next;
    float base_prob;
};

std::vector<Transition> dynamic_transitions;
std::map<std::string, std::map<std::string, float>> LEXICAL_VECTORS;
std::map<std::string, std::map<std::string, int>> raw_bigram_counts;

// State machine for Serial ingestion
enum SystemState { STATE_IDLE, STATE_RECEIVING_DATA };
SystemState current_state = STATE_IDLE;

// Safe Cosine Similarity Math
float computeCosineSimilarity(const std::map<std::string, float>& a, const std::map<std::string, float>& b) {
    if (a.empty() || b.empty()) return 0.0f;
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (const auto& pair : a) {
        norm_a += pair.second * pair.second;
        auto it = b.find(pair.first);
        if (it != b.end()) {
            dot += pair.second * it->second;
        }
    }
    for (const auto& pair : b) {
        norm_b += pair.second * pair.second;
    }

    if (norm_a <= 0.0f || norm_b <= 0.0f) return 0.0f;
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

// Sigmoid Curve Math
float sigmoidCurve(float value, float k, float midpoint) {
    return 1.0f / (1.0f + expf(-k * (value - midpoint)));
}

// Tokenizer
std::vector<std::string> tokenize(String text) {
    std::vector<std::string> tokens;
    int start = 0;
    int end = text.indexOf(' ');
    while (end != -1) {
        String token = text.substring(start, end);
        token.trim();
        if (token.length() > 0) tokens.push_back(token.c_str());
        start = end + 1;
        end = text.indexOf(' ', start);
    }
    String lastToken = text.substring(start);
    lastToken.trim();
    if (lastToken.length() > 0) tokens.push_back(lastToken.c_str());
    return tokens;
}

// Safe Ingestion Line Parser
void ingestTextLine(String line) {
    std::vector<std::string> words = tokenize(line);
    if (words.size() < 2) return;

    for (size_t i = 0; i < words.size() - 1; ++i) {
        std::string prev = words[i];
        std::string next = words[i+1];
        
        raw_bigram_counts[prev][next]++;
        
        // Populate lexical vectors safely
        LEXICAL_VECTORS[prev]["ctx_" + prev] = 0.9f;
        LEXICAL_VECTORS[prev]["ctx_" + next] = 0.4f;
        LEXICAL_VECTORS[next]["ctx_" + next] = 0.9f;
    }
}

// Finalize dataset
void finalizeDataset() {
    dynamic_transitions.clear();
    
    for (const auto& outer : raw_bigram_counts) {
        std::string prev = outer.first;
        int total = 0;
        for (const auto& inner : outer.second) {
            total += inner.second;
        }

        if (total > 0) {
            for (const auto& inner : outer.second) {
                float prob = (float)inner.second / (float)total;
                dynamic_transitions.push_back({prev, inner.first, prob});
            }
        }
    }
    Serial.printf("[Dataset] Finalized. Loaded %d transitions across %d unique contexts.\n", 
                  (int)dynamic_transitions.size(), (int)raw_bigram_counts.size());
}

// Real-Time Math Inference Loop with Safety Guards
void processPromptWithMath(String inputPrompt) {
    if (dynamic_transitions.empty()) {
        Serial.println("[Error] Dataset is empty! Send 'UPLOAD_START' and provide text first.");
        return;
    }

    std::vector<std::string> tokens = tokenize(inputPrompt);
    if (tokens.empty()) return;

    std::string currentContext = tokens.back();
    std::string generatedOutput = inputPrompt.c_str();

    Serial.println("\n--- Real-Time Mathematical Inference ---");
    Serial.printf("Input Token Context: %s\n", currentContext.c_str());

    for (int step = 0; step < 6; step++) {
        std::string bestNextToken = "<eos>";
        float maxScore = -1e9f;

        // Safely check if currentContext has an associated lexical vector
        std::map<std::string, float> sourceVector;
        auto vecIt = LEXICAL_VECTORS.find(currentContext);
        if (vecIt != LEXICAL_VECTORS.end()) {
            sourceVector = vecIt->second;
        }

        bool foundValidTransition = false;
        for (const auto& tx : dynamic_transitions) {
            if (tx.prev == currentContext) {
                foundValidTransition = true;
                std::map<std::string, float> targetVector;
                auto targetVecIt = LEXICAL_VECTORS.find(tx.next);
                if (targetVecIt != LEXICAL_VECTORS.end()) {
                    targetVector = targetVecIt->second;
                }
                
                float similarity = computeCosineSimilarity(sourceVector, targetVector);
                float curveWeight = sigmoidCurve(tx.base_prob, CURVE_K, CURVE_MIDPOINT);
                float score = logf(fmaxf(tx.base_prob, 1e-12f)) + (curveWeight * 0.5f * similarity);

                // Safe random noise generation for ESP32
                float thermalNoise = ((float)esp_random() / (float)UINT32_MAX) * 0.1f;
                score += thermalNoise;

                if (score > maxScore) {
                    maxScore = score;
                    bestNextToken = tx.next;
                }
            }
        }

        if (!foundValidTransition || bestNextToken == "<eos>") break;

        generatedOutput += " " + bestNextToken;
        currentContext = bestNextToken;
    }

    Serial.printf("Generated Output: %s\n", generatedOutput.c_str());
    Serial.println("----------------------------------------\n");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n[ESP32-C3] Safe Dynamic Dataset Engine Ready.");
    Serial.println("Commands:");
    Serial.println("1. Type 'UPLOAD_START' then paste text lines.");
    Serial.println("2. Type 'UPLOAD_END' to compile dataset.");
    Serial.println("3. Type any prompt to run inference.\n");
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        if (input.length() == 0) return;

        if (input == "UPLOAD_START") {
            current_state = STATE_RECEIVING_DATA;
            raw_bigram_counts.clear();
            LEXICAL_VECTORS.clear();
            dynamic_transitions.clear();
            Serial.println("[System] Ready for dataset text input... Send lines now.");
            return;
        }

        if (input == "UPLOAD_END") {
            current_state = STATE_IDLE;
            finalizeDataset();
            return;
        }

        if (current_state == STATE_RECEIVING_DATA) {
            ingestTextLine(input);
            Serial.printf("[Ingested] %s\n", input.c_str());
        } else {
            processPromptWithMath(input);
        }
    }
}
