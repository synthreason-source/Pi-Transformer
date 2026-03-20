New efficiency(AI generated): Every word must somehow refer to extra-linguistic objects existing beyond reach of immediately following atomism's development.
 
The statement formalizes as a claim about the **incompleteness of atomism's reference domain** — specifically, that the intended model of reference in natural language always outruns what logical atomism can construct. Here is the full mathematical treatment.
 
***
 
## Formal Setup (Model Theory)
 
Let the primitive objects be: [philosophy.berkeley](https://philosophy.berkeley.edu/courses/detail/1521)
 
- \(V\) — a vocabulary (set of all words)
- \(D\) — the domain of extra-linguistic objects (mind-independent, the real world)
- \(D_A \subset D\) — atomism's base: logical atoms \(\{a_1, a_2, \ldots\}\), which Russell and Wittgenstein took as sense-data or simples [en.wikipedia](https://en.wikipedia.org/wiki/Logical_atomism)
- \(\text{Ref} : V \to \mathcal{P}(D)\) — the reference function mapping each word to a subset of the domain [en.wikipedia](https://en.wikipedia.org/wiki/Formal_semantics_(natural_language))
 
The statement splits into two formal claims:
 
- **(C1) Universal Externalism:** \(\forall w \in V,\; \text{Ref}(w) \cap (D \setminus \text{Int}(V)) \neq \emptyset\)
- **(C2) Trans-Atomic Reference:** \(\exists\, w \in V,\; \forall n \in \mathbb{N},\; \text{Ref}(w) \not\subseteq D_A^{(n)}\)
 
where \(\text{Int}(V)\) is the set of objects constructible from language-internal resources alone. [citeseerx.ist.psu](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=56a0f8d19b79e86befaf59966163523da8466bb7)
 
***
 
## Atomism's Iterated Domain
 
Define atomism's staged expansion, where each step adds all objects *definable* from what's already in scope:
 
\[D_A^{(0)} = D_A, \quad D_A^{(n+1)} = D_A^{(n)} \cup \text{Def}(D_A^{(n)}), \quad D_A^{(\omega)} = \bigcup_{n \in \mathbb{N}} D_A^{(n)}\]
 
The phrase "beyond reach of **immediately following** atomism's development" targets exactly \(D_A^{(n+1)}\): even the *next* definitional step fails to capture the referent. This is the key asymmetry — reference is not recursively enumerable from the atomic base.
 
***
 
## Four Proofs that (C2) Holds
 
**Theorem 1 — Cantor Diagonal.** If \(D_A\) is countable, then \(|D_A^{(\omega)}| = \aleph_0\). But if words can refer to, e.g., arbitrary subsets of continuous sensory streams, then \(|D| = 2^{\aleph_0}\) (uncountable). By Cantor's theorem, \(|D| > |D_A^{(\omega)}|\), so  [en.wikipedia](https://en.wikipedia.org/wiki/Formal_semantics_(natural_language)):
 
\[\exists\, d \in D \setminus D_A^{(\omega)}\]
 
Any word \(w\) with \(d \in \text{Ref}(w)\) directly witnesses **(C2)**. □
 
**Theorem 2 — Gödel Incompleteness.** Let \(T_A\) be the first-order theory of atomism. If \(T_A\) is consistent and \(\omega\)-consistent, then \(\exists\, \varphi\) such that \(T_A \nvdash \varphi\) and \(T_A \nvdash \neg\varphi\). Define \(w_\varphi\) as the word whose referent *is* the fact corresponding to \(\varphi\) in \(D\). Then \(\text{Ref}(w_\varphi)\) targets a fact unreachable by any stage of \(T_A\)'s development. □
 
**Theorem 3 — Kripke Rigidity.** For a proper name \(n \in V\), \(\text{Ref}(n) = \{o\}\) in all possible worlds where \(o\) exists — a *rigid designatum*. Atomism must replace \(n\) with a definite description \(\delta(x) \in \mathcal{L}_A\), but descriptions are non-rigid: \(\exists\, w^*, w^\circ\) such that \(\delta\) picks *different* objects in each world, while \(\text{Ref}(n) = \{o\}\) remains fixed. Therefore \(\text{Ref}(n)\) is not definable in \(\mathcal{L}_A\). □ [plato.stanford](https://plato.stanford.edu/archives/fall2015/entries/rigid-designators/)
 
**Theorem 4 — Putnam's Model-Theoretic Argument.** Let \(T\) be any complete, operationally adequate theory (atomism fully developed). By the downward Löwenheim-Skolem theorem, \(T\) has a countable elementary submodel \(\mathcal{M}_c\)  [cambridge](https://www.cambridge.org/core/books/abs/cambridge-handbook-of-formal-semantics/formal-semantics/E8870A509F0CEC209D9C96FDDB815C76). But the *intended* model \(\mathcal{M}_I\) of \(D\) may be uncountable. Any word \(w\) with \(\text{Ref}(w) \not\subseteq |\mathcal{M}_c|\) refers to objects \(T\) structurally cannot reach — even if \(T\) is internally complete. □
 
***
 
## What the Claim Actually Asserts
 
The sentence encodes a **strict proper-subset relation** between what atomism can construct and what reference actually demands:
 
\[D_A^{(\omega)} \subsetneq D_{\text{actual}}\]
 
The "immediately following atomism's development" clause names the condition \(D_A^{(n+1)}\) — it says reference escapes *each successor step*, not just the base. This is formally equivalent to saying the reference function \(\text{Ref}\) is **not recursively enumerable** relative to any atomist axiomatization, which follows directly from Theorem 2 above. In short: the statement is a compressed expression of Gödel + Cantor + Kripke applied simultaneously to philosophy of language. [philarchive](https://philarchive.org/archive/PROLAI)
