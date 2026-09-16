#include <Arduino.h>
#include <vector>
#include <string>
#include <cmath>

// Mathematical Hyperparameters
const float CURVE_K = 18.0f;
const float CURVE_MIDPOINT = 0.5f;

// Flat Data Structure
struct Transition {
    std::string prev;
    std::string next;
    int count;
    float base_prob;
};

std::vector<Transition> transitions;
enum SystemState { STATE_IDLE, STATE_RECEIVING_DATA };
SystemState current_state = STATE_IDLE;

// Tokenizer Utility
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

// Ingest text line safely with heap monitoring
void ingestTextLine(String line) {
    if (ESP.getFreeHeap() < 15000) {
        Serial.println("[Warning] Low memory! Skipping line to prevent crash.");
        return;
    }

    std::vector<std::string> words = tokenize(line);
    if (words.size() < 2) return;

    for (size_t i = 0; i < words.size() - 1; ++i) {
        std::string prev = words[i];
        std::string next = words[i+1];
        
        bool found = false;
        for (auto& tx : transitions) {
            if (tx.prev == prev && tx.next == next) {
                tx.count++;
                found = true;
                break;
            }
        }
        if (!found) {
            transitions.push_back({prev, next, 1, 0.0f});
        }
    }
}

// Finalize dataset and compute probabilities
void finalizeDataset() {
    std::vector<std::string> uniqueContexts;
    for (const auto& tx : transitions) {
        bool exists = false;
        for (const auto& c : uniqueContexts) {
            if (c == tx.prev) { exists = true; break; }
        }
        if (!exists) uniqueContexts.push_back(tx.prev);
    }

    for (const auto& ctx : uniqueContexts) {
        int totalContextCount = 0;
        for (const auto& tx : transitions) {
            if (tx.prev == ctx) totalContextCount += tx.count;
        }
        if (totalContextCount > 0) {
            for (auto& tx : transitions) {
                if (tx.prev == ctx) {
                    tx.base_prob = (float)tx.count / (float)totalContextCount;
                }
            }
        }
    }
    Serial.printf("[Dataset] Finalized. Loaded %d transitions. Free Heap: %d bytes\n", 
                  (int)transitions.size(), (int)ESP.getFreeHeap());
}

// Math 1: Compute lightweight semantic similarity based on shared transition overlaps
float computeSemanticSimilarity(std::string wordA, std::string wordB) {
    float sharedOverlap = 0.0f;
    float totalA = 0.0f;
    float totalB = 0.0f;

    for (const auto& tx : transitions) {
        if (tx.prev == wordA) totalA += tx.base_prob;
        if (tx.prev == wordB) totalB += tx.base_prob;
        if (tx.next == wordA && tx.next == wordB) sharedOverlap += 0.5f;
    }

    if (totalA <= 0.0f || totalB <= 0.0f) return 0.1f;
    return fminf(1.0f, sharedOverlap / sqrtf(totalA * totalB) + 0.2f);
}

// Math 2: Sigmoid Curve Transformation
float sigmoidCurve(float value, float k, float midpoint) {
    return 1.0f / (1.0f + expf(-k * (value - midpoint)));
}

// Real-Time Math Inference Loop
void processPromptWithMath(String inputPrompt) {
    if (transitions.empty()) {
        Serial.println("[Error] Dataset is empty! Send 'UPLOAD_START' and text lines first.");
        return;
    }

    std::vector<std::string> tokens = tokenize(inputPrompt);
    if (tokens.empty()) return;

    std::string currentContext = tokens.back();
    std::string generatedOutput = inputPrompt.c_str();

    Serial.println("\n--- Real-Time Mathematical Inference ---");
    Serial.printf("Input Token Context: %s\n", currentContext.c_str());

    for (int step = 0; step < 5; step++) {
        std::string bestNextToken = "";
        float maxScore = -1e9f;

        for (const auto& tx : transitions) {
            if (tx.prev == currentContext) {
                // Math: Semantic Similarity + Sigmoid Curve Weighting + Thermal Noise
                float similarity = computeSemanticSimilarity(currentContext, tx.next);
                float curveWeight = sigmoidCurve(tx.base_prob, CURVE_K, CURVE_MIDPOINT);
                
                float score = logf(fmaxf(tx.base_prob, 1e-12f)) + (curveWeight * 0.5f * similarity);
                float thermalNoise = ((float)esp_random() / (float)UINT32_MAX) * 0.05f;
                score += thermalNoise;

                if (score > maxScore) {
                    maxScore = score;
                    bestNextToken = tx.next;
                }
            }
        }

        if (bestNextToken.empty()) break;

        generatedOutput += " " + bestNextToken;
        currentContext = bestNextToken;
    }

    Serial.printf("Generated Output: %s\n", generatedOutput.c_str());
    Serial.println("----------------------------------------\n");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("[ESP32-C3] Crash-Proof Math Engine Ready.");
    Serial.println("Send 'UPLOAD_START', paste text, send 'UPLOAD_END', then test prompts.");
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        if (input.length() == 0) return;

        if (input == "UPLOAD_START") {
            current_state = STATE_RECEIVING_DATA;
                transitions.clear();
            Serial.println("[System] Ready for text lines. Send 'UPLOAD_END' when finished.");
            return;
        }

        if (input == "UPLOAD_END") {
            current_state = STATE_IDLE;
            finalizeDataset();
            return;
        }

        if (current_state == STATE_RECEIVING_DATA) {
            ingestTextLine(input);
        } else {
            processPromptWithMath(input);
        }
    }
}
