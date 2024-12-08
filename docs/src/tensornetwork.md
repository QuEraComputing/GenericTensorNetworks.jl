# An introduction to tensor networks

Let $G = (V, E)$ be a hypergraph, where $V$ is the set of vertices and $E$ is the set of hyperedges. Each vertex $v \in V$ is associated with a local variable, e.g. "spin" and "bit". A hyperedge $e \in E$ is a subset of vertices $e \subseteq V$. On top of which, we can define a local Hamiltonian $H$ as a sum of local terms $h_e$ over all hyperedges $e \in E$:

```math
H(\sigma) = \sum_{e \in E} h_e(\sigma_e)
```

where $\sigma_e$ is the restriction of the configuration $\sigma$ to the vertices in $e$.

The following solution space properties are of interest:

* The partition function,
    ```math
    Z = \sum_{\sigma} e^{-\beta H(\sigma)}
    ```
    where $\beta$ is the inverse temperature.
* The maximum/minimum solution sizes,
    ```math
    \max_{\sigma} H(\sigma), \min_{\sigma} H(\sigma)
    ```
* The number of solutions at certain sizes,
    ```math
    N(k) = \sum_{\sigma} \delta(k, H(\sigma))
    ```
* The enumeration of solutions at certain sizes.
    ```math
    S = \{ \sigma | H(\sigma) = k \}
    ```
* The direct sampling of solutions at certain sizes.
    ```math
    \sigma \sim S
    ```

## Tensor network representation

### Partition function
It is well known that the partition function of an energy model can be represented as a tensor network[^Levin2007]. The partition function can be written in a sum-product form as
```math
Z = \sum_{\sigma} e^{-\beta H(\sigma)} = \sum_{\sigma} \prod_{e \in E} T_e(\sigma_e)
```
where $T_e(\sigma_e) = e^{-\beta h_e(\sigma_e)}$ is a tensor associated with the hyperedge $e$.

This sum-product form is directly related to a tensor network $(V, \{T_{\sigma_e} \mid e\in E\}, \emptyset)$, where $T_{\sigma_e}$ is a tensor labeled by $\sigma_e \subseteq V$, and its elements are defined by $T_{\sigma_e}= T_e(\sigma_e)$. $\emptyset$ is the set of open vertices in a tensor network, which are not summed over.

### Maximum/minimum solution sizes
The maximum/minimum solution sizes can be represented as a tensor network as well. The maximum solution size can be written as
```math
\max_{\sigma} H(\sigma) = \max_{\sigma} \sum_{e \in E} h_e(\sigma_e)
```
which can be represented as a tropical tensor network[^Liu2021] $(V, \{h_{\sigma_e} \mid e\in E\}, \emptyset)$, where $h_{\sigma_e}$ is a tensor labeled by $\sigma_e \subseteq V$, and its elements are defined by $h_{\sigma_e}= h_e(\sigma_e)$.

## Problems
### Independent set problem
The independent set problem on graph $G=(V, E)$ is characterized by the Hamiltonian
```math
H(\sigma) = U \sum_{(i, j) \in E}  n_i n_j - \sum_{i \in V} n_i
```
where $n_i \in \{0, 1\}$ is a binary variable associated with vertex $i$, and $U\rightarrow \infty$ is a large constant. The goal is to find the maximum independent set, i.e. the maximum number of vertices such that no two vertices are connected by an edge.
The partition function for an independent set problem is
```math
Z = \sum_{\sigma} e^{-\beta H(\sigma)} = \sum_{\sigma} \prod_{(i, j) \in E} e^{-\beta U n_in_j} \prod_{i \in V} e^{\beta n_i}
```

Let $x = e^{\beta}$, the partition function can be written as
```math
Z = \sum_{\sigma} \prod_{(i, j) \in E} B_{n_in_j} \prod_{i \in V} W_{n_i}
```
where $B_{n_in_j} = \lim_{U \rightarrow \infty} e^{-U \beta n_in_j}=\begin{cases}0, \quad n_in_j = 1\\1,\quad n_in_j = 0\end{cases}$ and $W_{n_i} = x^{n_i}$ are tensors associated with the hyperedge $(i, j)$ and the vertex $i$, respectively.

The tensor network representation for the partition function is
```math
\mathcal{N}_{IS} = (\Lambda, \{B_{n_in_j} \mid (i, j)\in E\} \cup \{W_{n_i} \mid i\in \Lambda\}, \emptyset)
```
where $\Lambda = \{n_i \mid i \in V\}$ is the set of binary variables, $B_{n_in_j}$ is a tensor associated with the hyperedge $(i, j)$ and $W_{n_i}$ is a tensor associated with the vertex $i$. The tensors are defined as
```math
W = \left(\begin{matrix}
    1 \\
    x
    \end{matrix}\right)
```
where $x$ is a variable associated with $v$.
```math
B = \left(\begin{matrix}
1  & 1\\
1 & 0
\end{matrix}\right).
```

The contraction of the tensor network $\mathcal{N}_{IS}$ gives the partition function $Z$. It is implicitly assumed that the tensor elements are real numbers.

However, by replacing the tensor elements with tropical numbers, the tensor network $\mathcal{N}_{IS}$ can be used to compute the maximum independent set size and its degeneracy[^Liu2021].

An algebra can be defined by
```math
\begin{align*}
\oplus &= \max\\
\otimes &= +
\end{align*}
```

[^Levin2007]: Levin, M., Nave, C.P., 2007. Tensor renormalization group approach to two-dimensional classical lattice models. Physical Review Letters 99, 1–4. https://doi.org/10.1103/PhysRevLett.99.120601
[^Liu2021]: Liu, J.-G., Wang, L., Zhang, P., 2021. Tropical Tensor Network for Ground States of Spin Glasses. Phys. Rev. Lett. 126, 090506. https://doi.org/10.1103/PhysRevLett.126.090506