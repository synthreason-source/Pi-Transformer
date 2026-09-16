#include <Arduino.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <sstream>

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

// Cosine Similarity Math
float computeCosineSimilarity(const std::map<std::string, float>& a, const std::map<std::string, float>& b) {
    if (a.empty() || b.empty()) return 0.0f;
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (const auto& [key, val] : a) {
        norm_a += val * val;
        auto it = b.find(key);
        if (it != b.end()) dot += val * it->second;
    }
    for (const auto& [key, val] : b) norm_b += val * val;

    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

// Sigmoid Curve Math
float sigmoidCurve(float value, float k, float midpoint) {
    return 1.0f / (1.0f + exp(-k * (value - midpoint)));
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

// Ingest a line of text into bigram counts and vectors
void ingestTextLine(String line) {
    std::vector<std::string> words = tokenize(line);
    if (words.size() < 2) return;

    for (size_t i = 0; i < words.size() - 1; ++i) {
        std::string prev = words[i];
        std::string next = words[i+1];
        
        raw_bigram_counts[prev][next]++;
        
        // Build mock lexical sparse vectors dynamically based on context occurrence
        LEXICAL_VECTORS[prev]["ctx_" + prev] = 0.9f;
        LEXICAL_VECTORS[prev]["ctx_" + next] = 0.4f;
        LEXICAL_VECTORS[next]["ctx_" + next] = 0.9f;
    }
}

// Finalize dataset by converting raw counts into probabilities
void finalizeDataset() {
    dynamic_transitions.clear();
    
    for (const auto& [prev, next_counts] : raw_bigram_counts) {
        int total = 0;
        for (const auto& [next, count] : next_counts) {
            total += count;
        }

        for (const auto& [next, count] : next_counts) {
            float prob = (float)count / total;
            dynamic_transitions.push_back({prev, next, prob});
        }
    }
    Serial.printf("[Dataset] Finalized. Loaded %d transitions across %d unique contexts.\n", 
                  dynamic_transitions.size(), raw_bigram_counts.size());
}

// Real-Time Math Inference Loop using Dynamic Dataset
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

    std::map<std::string, float> sourceVector = LEXICAL_VECTORS[currentContext];

    for (int step = 0; step < 6; step++) {
        std::string bestNextToken = "<eos>";
        float maxScore = -1e9f;

        for (const auto& tx : dynamic_transitions) {
            if (tx.prev == currentContext) {
                std::map<std::string, float> targetVector = LEXICAL_VECTORS[tx.next];
                
                float similarity = computeCosineSimilarity(sourceVector, targetVector);
                float curveWeight = sigmoidCurve(tx.base_prob, CURVE_K, CURVE_MIDPOINT);
                float score = log(fmax(tx.base_prob, 1e-12f)) + (curveWeight * 0.5f * similarity);

                float thermalNoise = ((float)esp_random() / UINT32_MAX) * 0.1f;
                score += thermalNoise;

                if (score > maxScore) {
                    maxScore = score;
                    bestNextToken = tx.next;
                }
            }
        }

        if (bestNextToken == "<eos>") break;

        generatedOutput += " " + bestNextToken;
        currentContext = bestNextToken;
        sourceVector = LEXICAL_VECTORS[currentContext];
    }

    Serial.printf("Generated Output: %s\n", generatedOutput.c_str());
    Serial.println("----------------------------------------\n");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n[ESP32-C3] Dynamic Dataset Engine Ready.");
    Serial.println("Commands:");
    Serial.println("1. Type 'UPLOAD_START' then paste text lines.");
    Serial.println("2. Type 'UPLOAD_END' to compile dataset.");
    Serial.println("3. Type any prompt (e.g. 'camera') to run inference.\n");
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
