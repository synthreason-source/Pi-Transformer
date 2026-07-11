import numpy as np
import re


def exp_sin(x, scale, freq):
    """
    Winning curve family from the stacking-curve selection charts.
    Assumed form: exp(scale * x) * (1 + sin(freq * x))
    Applied element-wise to whatever matrix x is.
    """
    return np.exp(scale * x) * (1 + np.sin(freq * x))


class InfluenceSpaceMarkov:

    def __init__(
        self,
        beta=2.0,
        alpha=3.0
    ):
        self.beta = beta
        self.alpha = alpha


    def fit(self, text):

        # ---------------------
        # SENTENCE SPLIT
        # ---------------------

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text.strip()
        )

        sentence_words = [
            s.lower().split()
            for s in sentences
            if s.strip()
        ]

        self.sentences = sentence_words


        words = [
            w
            for s in sentence_words
            for w in s
        ]


        self.vocab = sorted(
            set(words)
        )


        self.word_to_idx = {
            w:i
            for i,w in enumerate(self.vocab)
        }


        n=len(self.vocab)



        # ---------------------
        # MARKOV MATRIX
        # ---------------------

        counts=np.zeros(
            (n,n)
        )


        for sent in sentence_words:

            for a,b in zip(
                sent[:-1],
                sent[1:]
            ):

                i=self.word_to_idx[a]
                j=self.word_to_idx[b]

                counts[i,j]+=1


        self.counts=counts



        # ---------------------
        # INFLUENCE FIELD
        # ---------------------
        # Stage 1 winner: exp_sin, scale=4.0, freq=0.5
        # x = log1p(counts)   (matches stage_1_L_to_Y chart domain 0 -> ~3.6)

        x1 = np.log1p(counts)

        Y = exp_sin(x1, scale=4.0, freq=0.5)

        Y[counts==0]=0

        self.Y=Y



        ymax=max(
            Y.max(),
            1
        )


        # ---------------------
        # Stage 2 winner: exp_sin, scale=4.0, freq=2.0
        # x = Y / ymax   (matches stage_2_Y_to_Z chart domain 0 -> 1.0)

        x2 = Y / ymax

        Z = exp_sin(x2, scale=4.0, freq=2.0)

        Z[Y==0]=0

        self.Z=Z


        sums=Z.sum(
            axis=1,
            keepdims=True
        )


        self.P=np.divide(
            Z,
            sums,
            out=np.zeros_like(Z),
            where=sums!=0
        )



        # ---------------------
        # CLASSICAL CO-OCCURRENCE
        # ---------------------

        window=4

        cooc=np.zeros(
            (n,n)
        )


        for sent in sentence_words:

            for i,w in enumerate(sent):

                wi=self.word_to_idx[w]

                left=max(
                    0,
                    i-window
                )

                right=min(
                    len(sent),
                    i+window+1
                )


                for j in range(left,right):

                    if i==j:
                        continue

                    wj=self.word_to_idx[
                        sent[j]
                    ]

                    cooc[wi,wj]+=1



        self.cooc=cooc



        # ---------------------
        # COSINE SPACE
        # ---------------------

        norms=np.linalg.norm(
            cooc,
            axis=1,
            keepdims=True
        )


        norms[norms==0]=1


        E=cooc/norms


        self.embedding=E


        self.cosine=(
            E @ E.T
        )


        return self



    # =================================================
    # SPACE A
    # INFLUENCE GENERATOR
    # =================================================

    def influence_generate(
        self,
        start,
        length=50
    ):

        start=start.lower()


        if start not in self.word_to_idx:

            start=np.random.choice(
                self.vocab
            )


        current=start

        result=[
            current
        ]


        for _ in range(length):

            i=self.word_to_idx[current]

            p=self.P[i]


            if p.sum()==0:
                break


            current=np.random.choice(
                self.vocab,
                p=p
            )


            result.append(
                current
            )


        return result



    # =================================================
    # SPACE B
    # COSINE WALK
    # =================================================

    def semantic_generate(
        self,
        start,
        length=50
    ):

        start=start.lower()


        if start not in self.word_to_idx:

            start=np.random.choice(
                self.vocab
            )


        current=start

        result=[
            current
        ]


        for _ in range(length):

            i=self.word_to_idx[current]


            sim=self.cosine[i].copy()


            sim[i]=0


            sim=np.maximum(
                sim,
                0
            )


            if sim.sum()==0:
                break


            sim/=sim.sum()


            current=np.random.choice(
                self.vocab,
                p=sim
            )


            result.append(
                current
            )


        return result



    # =================================================
    # VECTOR REPRESENTATION
    # =================================================

    def sentence_vector(
        self,
        words
    ):

        vectors=[]


        for w in words:

            if w in self.word_to_idx:

                i=self.word_to_idx[w]

                vectors.append(
                    self.embedding[i]
                )


        if not vectors:

            return np.zeros(
                len(self.vocab)
            )


        return np.mean(
            vectors,
            axis=0
        )



    def cosine_similarity(
        self,
        a,
        b
    ):

        denom=(
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )


        if denom==0:
            return 0


        return np.dot(a,b)/denom



    # =================================================
    # INTERSECTION SELECTOR
    # =================================================

    def intersection_generate(
        self,
        start,
        candidates=50,
        length=50
    ):


        # semantic target

        semantic_path=self.semantic_generate(
            start,
            length
        )


        semantic_vector=self.sentence_vector(
            semantic_path
        )


        best=None


        for _ in range(candidates):


            candidate=self.influence_generate(
                start,
                length
            )


            candidate_vector=self.sentence_vector(
                candidate
            )


            score=self.cosine_similarity(
                candidate_vector,
                semantic_vector
            )


            if best is None or score>best[0]:

                best=(
                    score,
                    candidate
                )


        return best



# =====================================================
# LOAD CORPUS
# =====================================================


with open(
    "singlekb.txt",
    "r",
    encoding="utf8"
) as f:

    corpus=f.read()



model=InfluenceSpaceMarkov(
    beta=2.0,
    alpha=3.0
)


model.fit(
    corpus
)



# =====================================================
# CHAT
# =====================================================


while True:

    prompt=input(
        "USER: "
    ).strip()


    if not prompt:
        continue


    seed=prompt.split()[-1]


    score,result=model.intersection_generate(
        seed,
        candidates=30,
        length=60
    )


    print()

    print(
        " ".join(result)
    )

    print(
        "-"*80
    )
