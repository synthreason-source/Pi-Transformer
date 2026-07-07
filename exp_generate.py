import numpy as np
import re


class InfluenceSpaceMarkov:

    def __init__(
        self,
        beta=2.0,
        alpha=3.0,
        gamma=2.0
    ):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma


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


        n = len(self.vocab)


        # ---------------------
        # MARKOV MATRIX
        # ---------------------

        counts = np.zeros(
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
        # SINE LOG INFLUENCE FIELD
        # ---------------------

        L=np.log1p(counts)


        Y=np.exp(
            self.beta * L
            +
            np.sin(
                self.gamma * L
            )
        )


        Y[counts==0]=0


        self.Y=Y



        # ---------------------
        # OSCILLATING ENERGY FIELD
        # ---------------------

        ymax=max(
            Y.max(),
            1
        )


        Yn=Y/ymax


        Z=np.exp(
            self.alpha*Yn
            +
            np.sin(
                self.gamma*Yn
            )
        )


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
        # CO-OCCURRENCE SPACE
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
        # EMBEDDING SPACE
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
    # INFLUENCE WALK
    # =================================================

    def influence_generate(
        self,
        start,
        length=50
    ):


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
    # VECTOR SPACE
    # =================================================

    def sentence_vector(
        self,
        words
    ):

        vectors=[]


        for w in words:

            if w in self.word_to_idx:

                vectors.append(
                    self.embedding[
                        self.word_to_idx[w]
                    ]
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

        d=(
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )


        if d==0:
            return 0


        return np.dot(a,b)/d



    # =================================================
    # INTERSECTION DYNAMICS
    # =================================================

    def intersection_generate(
        self,
        start,
        candidates=50,
        length=50
    ):


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
# PROMPT ENDOFUNCTION
# =====================================================

def prompt_endofunction(prompt):

    words=prompt.lower().split()


    if not words:
        return ""


    transformed=[]


    for w in words:

        phase=np.sin(
            len(w)
        )


        if phase >= 0:

            transformed.append(
                w
            )

        else:

            transformed.insert(
                0,
                w
            )


    # nonlinear collapse to a single attractor word

    index=int(
        abs(
            np.sin(
                len(prompt)
            )
        )
        *
        (len(transformed)-1)
    )


    return transformed[index]



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
    alpha=3.0,
    gamma=2.0
)


model.fit(
    corpus
)



# =====================================================
# CHAT LOOP
# =====================================================

while True:

    prompt=input(
        "USER: "
    ).strip()


    if not prompt:
        continue


    if prompt.lower() in [
        "exit",
        "quit"
    ]:
        break



    # prompt becomes its own transformed state

    seed=prompt_endofunction(
        prompt
    )


    score,result=model.intersection_generate(
        seed,
        candidates=30,
        length=600
    )


    print()

    print(
        " ".join(result)
    )

    print()

    print(
        "-"*80
    )
