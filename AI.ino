#include <Arduino.h>
#include <vector>
#include <string>
#include <cmath>

// Mathematical Hyperparameters
const float CURVE_K = 18.0f;
const float CURVE_MIDPOINT = 0.5f;

// Flat Data Structures
struct Transition {
    std::string prev;
    std::string next;
    int count;
    float base_prob;
};

struct LexicalFeature {
    std::string word;
    std::string feature_key;
    float weight;
};

std::vector<Transition> transitions;
std::vector<LexicalFeature> lexicon;

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

// Helper: Add or update a lexical feature weight safely
void setLexicalFeature(std::string word, std::string key, float val) {
    for (auto& lf : lexicon) {
        if (lf.word == word && lf.feature_key == key) {
            lf.weight = val;
            return;
        }
    }
    lexicon.push_back({word, key, val});
}

// Helper: Get all features for a specific word
std::vector<LexicalFeature> getWordFeatures(std::string word) {
    std::vector<LexicalFeature> result;
    for (const auto& lf : lexicon) {
        if (lf.word == word) {
            result.push_back(lf);
        }
    }
    return result;
}

// Math 1: Cosine Similarity between two words using flat feature vectors
float computeCosineSimilarity(std::string wordA, std::string wordB) {
    std::vector<LexicalFeature> vecA = getWordFeatures(wordA);
    std::vector<LexicalFeature> vecB = getWordFeatures(wordB);

    if (vecA.empty() || vecB.empty()) return 0.0f;

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (const auto& fa : vecA) {
        norm_a += fa.weight * fa.weight;
        for (const auto& fb : vecB) {
            if (fa.feature_key == fb.feature_key) {
                dot += fa.weight * fb.weight;
            }
        }
    }

    for (const auto& fb : vecB) {
        norm_b += fb.weight * fb.weight;
    }

    if (norm_a <= 0.0f || norm_b <= 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

// Math 2: Sigmoid Curve Transformation
float sigmoidCurve(float value, float k, float midpoint) {
    return 1.0f / (1.0f + expf(-k * (value - midpoint)));
}

// Ingest text line into transitions and dynamic lexical features
void ingestTextLine(String line) {
    std::vector<std::string> words = tokenize(line);
    if (words.size() < 2) return;

    for (size_t i = 0; i < words.size() - 1; ++i) {
        std::string prev = words[i];
        std::string next = words[i+1];
        
        // Track bigram counts
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

        // Build sparse lexical feature vectors dynamically
        setLexicalFeature(prev, "ctx_" + prev, 0.9f);
        setLexicalFeature(prev, "ctx_" + next, 0.4f);
        setLexicalFeature(next, "ctx_" + next, 0.9f);
    }
}

// Finalize dataset by computing base probabilities
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
    Serial.printf("[Dataset] Finalized. Loaded %d transitions, %d lexical features.\n", 
                  (int)transitions.size(), (int)lexicon.size());
}

// Real-Time Math Inference Loop (Cosine Similarity + Sigmoid + Noise)
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
        std::string bestNextToken = "<eos>";
        float maxScore = -1e9f;

        for (const auto& tx : transitions) {
            if (tx.prev == currentContext) {
                // Math: Calculate Vector Cosine Similarity & Sigmoid Curve Weighting
                float similarity = computeCosineSimilarity(currentContext, tx.next);
                float curveWeight = sigmoidCurve(tx.base_prob, CURVE_K, CURVE_MIDPOINT);
                
                // Combined scoring equation
                float score = logf(fmaxf(tx.base_prob, 1e-12f)) + (curveWeight * 0.5f * similarity);

                // Hardware thermal noise via ESP32 RNG
                float thermalNoise = ((float)esp_random() / (float)UINT32_MAX) * 0.05f;
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
    }

    Serial.printf("Generated Output: %s\n", generatedOutput.c_str());
    Serial.println("----------------------------------------\n");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("[ESP32-C3] Flat-Memory Math Engine Ready.");
    Serial.println("Commands:");
    Serial.println("1. Send 'UPLOAD_START' then paste text lines.");
    Serial.println("2. Send 'UPLOAD_END' to compile dataset.");
    Serial.println("3. Type any prompt to run math inference.\n");
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        if (input.length() == 0) return;

        if (input == "UPLOAD_START") {
            current_state = STATE_RECEIVING_DATA;
            transitions.clear();
            lexicon.clear();
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
            Serial.printf("[Ingested] %s\n", input.c_str());
        } else {
            processPromptWithMath(input);
        }
    }
}
