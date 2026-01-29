"""Technical depth contrast pairs for steering vector extraction.

These pairs contrast:
- Positive: Detailed, technical, expert-level responses
- Negative: Simplified, high-level, beginner-friendly responses

The goal is to extract a vector that controls technical depth in responses.
"""

from .base import ContrastPair, ContrastPairDataset

# Technical depth contrast pairs
# Positive = deep/technical, Negative = simplified/high-level
TECHNICAL_DEPTH_PAIRS = [
    # Programming concepts
    {
        "positive": "User: How does garbage collection work?\nAssistant: Garbage collection reclaims memory occupied by objects no longer reachable from the root set. Modern collectors like G1 use generational hypothesis - most objects die young - partitioning the heap into young (Eden, Survivor spaces) and old generations. Minor GCs use copying collection in young gen with stop-the-world pauses, while major GCs employ mark-sweep-compact or concurrent marking (CMS, ZGC) to minimize latency. G1 divides heap into regions, collecting garbage-first regions with highest reclamation potential. Write barriers track cross-region references in remembered sets.",
        "negative": "User: How does garbage collection work?\nAssistant: Garbage collection automatically frees up memory that your program is no longer using. It finds objects that can't be accessed anymore and removes them, so you don't have to manually manage memory.",
    },
    {
        "positive": "User: What is a hash table?\nAssistant: A hash table is an associative array providing O(1) average-case lookup via a hash function that maps keys to bucket indices. Collision resolution strategies include separate chaining (linked lists per bucket) and open addressing (linear/quadratic probing, double hashing). Load factor α = n/m determines performance; rehashing occurs when α exceeds threshold. Perfect hash functions eliminate collisions for static sets. Cryptographic applications use collision-resistant functions like SHA-256. Robin Hood hashing improves cache performance by limiting probe sequence variance.",
        "negative": "User: What is a hash table?\nAssistant: A hash table is a data structure that stores key-value pairs. It uses a special function to quickly find where each item is stored, making lookups very fast.",
    },
    {
        "positive": "User: Explain TCP.\nAssistant: TCP is a connection-oriented, reliable transport protocol implementing full-duplex byte streams. Connection establishment uses three-way handshake (SYN, SYN-ACK, ACK) with ISN negotiation. Reliability comes from sequence numbers, cumulative ACKs, and retransmission timers (RTO calculated via Jacobson/Karels algorithm). Flow control uses sliding window with receiver-advertised rwnd. Congestion control implements AIMD via slow start, congestion avoidance, fast retransmit, and fast recovery (Reno/CUBIC/BBR). Selective ACKs (SACK) improve loss recovery. Nagle's algorithm reduces small packet overhead; delayed ACKs batch acknowledgments.",
        "negative": "User: Explain TCP.\nAssistant: TCP is a protocol that ensures reliable communication between computers over the internet. It makes sure all data arrives correctly and in order, handling any lost packets automatically.",
    },
    {
        "positive": "User: How do neural networks learn?\nAssistant: Neural networks learn via gradient-based optimization of a loss function. Forward propagation computes activations through weighted sums and nonlinear activations (ReLU, GELU, softmax). Backpropagation applies chain rule to compute ∂L/∂w for each parameter. SGD updates: w ← w - η∇L, with momentum accumulating gradient history. Adam combines momentum with adaptive per-parameter learning rates via first/second moment estimates. Batch normalization stabilizes training by normalizing layer inputs. Regularization (L2, dropout, weight decay) prevents overfitting. Modern architectures use residual connections, attention mechanisms, and layer normalization.",
        "negative": "User: How do neural networks learn?\nAssistant: Neural networks learn by adjusting their internal connections based on examples. They make predictions, compare them to correct answers, and gradually improve by tweaking their settings to reduce errors.",
    },
    {
        "positive": "User: What is a database index?\nAssistant: Database indexes are auxiliary data structures accelerating query execution by avoiding full table scans. B+ trees are standard, maintaining sorted keys in balanced tree with O(log n) search; leaf nodes form linked list for range queries. Hash indexes provide O(1) equality lookups. Composite indexes follow leftmost prefix rule. Covering indexes include all query columns, enabling index-only scans. Partial indexes filter rows; expression indexes index computed values. Index maintenance overhead affects write performance. Query planners use statistics (histograms, cardinality estimates) for cost-based index selection. Clustered indexes determine physical row order.",
        "negative": "User: What is a database index?\nAssistant: A database index is like a book's index - it helps the database find information quickly without reading every row. It makes searches faster but takes up extra space.",
    },
    # Systems concepts
    {
        "positive": "User: How does virtual memory work?\nAssistant: Virtual memory provides process isolation and extends physical memory via demand paging. The MMU translates virtual addresses through multi-level page tables (PGD→PUD→PMD→PTE on x86-64). TLBs cache translations; misses trigger page table walks. Page faults invoke the kernel: minor faults for present-but-unmapped pages, major faults require disk I/O. Copy-on-write optimizes fork(). Page replacement algorithms (LRU approximations like clock) select eviction victims. Huge pages (2MB/1GB) reduce TLB pressure. ASLR randomizes mappings for security. Memory-mapped files enable zero-copy I/O.",
        "negative": "User: How does virtual memory work?\nAssistant: Virtual memory lets your computer use hard drive space as extra RAM. Each program thinks it has its own private memory, and the system handles moving data between RAM and disk as needed.",
    },
    {
        "positive": "User: Explain HTTPS.\nAssistant: HTTPS layers HTTP over TLS, providing confidentiality, integrity, and authentication. TLS 1.3 handshake: client sends supported cipher suites and key share (ECDHE); server responds with certificate chain, selects cipher, and completes key exchange. Certificates are verified against trusted CA roots using X.509 chain validation and revocation checking (OCSP/CRL). Symmetric session keys derive from shared secret via HKDF. AEAD ciphers (AES-GCM, ChaCha20-Poly1305) encrypt records with authenticated encryption. Forward secrecy ensures past session security despite future key compromise. Certificate Transparency logs provide public auditability.",
        "negative": "User: Explain HTTPS.\nAssistant: HTTPS is the secure version of HTTP. It encrypts the data between your browser and websites so others can't read it. The padlock icon shows when a connection is secure.",
    },
    {
        "positive": "User: How does Docker work?\nAssistant: Docker leverages Linux kernel features for OS-level virtualization. Namespaces provide isolation: PID (process tree), NET (network stack), MNT (filesystem), UTS (hostname), IPC, USER. Cgroups limit and account for CPU, memory, I/O resources. Union filesystems (OverlayFS) layer images efficiently; copy-on-write minimizes storage. Images are content-addressable layers identified by SHA256 digests. Containers are isolated processes with their own root filesystem and network namespace. containerd manages container lifecycle; runc implements OCI runtime spec. Bridge networking uses veth pairs and iptables NAT for container connectivity.",
        "negative": "User: How does Docker work?\nAssistant: Docker packages applications with everything they need to run into containers. These containers are lightweight and isolated, making it easy to run apps consistently across different computers.",
    },
    {
        "positive": "User: What is a mutex?\nAssistant: A mutex (mutual exclusion) is a synchronization primitive ensuring only one thread accesses a critical section. Implementation typically uses atomic compare-and-swap (CAS) operations on a lock word. Spinning (spinlock) wastes CPU but avoids context switch overhead; blocking mutexes use futex syscalls to sleep waiters efficiently. Priority inversion occurs when high-priority threads wait for low-priority lock holders; priority inheritance protocols mitigate this. Recursive mutexes allow same-thread re-acquisition. Reader-writer locks permit concurrent reads. Lock-free algorithms using CAS avoid mutexes entirely but require careful memory ordering (acquire/release semantics).",
        "negative": "User: What is a mutex?\nAssistant: A mutex is a lock that prevents multiple threads from accessing the same resource at the same time. When one thread has the lock, others must wait their turn.",
    },
    {
        "positive": "User: How does DNS work?\nAssistant: DNS is a hierarchical distributed database mapping domain names to IP addresses. Resolution proceeds recursively or iteratively: resolver queries root servers (.root-servers.net), receives referral to TLD servers (.com), then authoritative servers for the domain. Records types include A (IPv4), AAAA (IPv6), CNAME (alias), MX (mail), NS (nameserver), TXT, SRV. TTL controls caching duration. DNSSEC adds cryptographic signatures (RRSIG records) verified via chain of trust from root KSK. DNS-over-HTTPS/TLS encrypts queries for privacy. Anycast routing provides global distribution and DDoS resilience.",
        "negative": "User: How does DNS work?\nAssistant: DNS translates website names like google.com into IP addresses that computers use. When you visit a website, your computer asks DNS servers to look up the address.",
    },
    # Algorithms
    {
        "positive": "User: Explain quicksort.\nAssistant: Quicksort is a divide-and-conquer algorithm with O(n log n) average-case complexity. Partitioning selects a pivot and rearranges elements: smaller values left, larger right. Lomuto partition scans linearly; Hoare's uses bidirectional pointers and fewer swaps. Pivot selection affects performance: median-of-three mitigates worst-case O(n²) on sorted input. Tail recursion optimization bounds stack depth to O(log n). Introsort hybridizes with heapsort when recursion exceeds 2⌊log₂n⌋. Three-way partitioning (Dutch national flag) handles duplicates efficiently. Cache-oblivious variants optimize memory hierarchy performance.",
        "negative": "User: Explain quicksort.\nAssistant: Quicksort is a fast sorting algorithm. It picks a pivot element, puts smaller items on one side and larger on the other, then repeats for each side until everything is sorted.",
    },
    {
        "positive": "User: What is dynamic programming?\nAssistant: Dynamic programming solves problems exhibiting optimal substructure and overlapping subproblems by caching intermediate results. Bottom-up (tabulation) builds solutions iteratively; top-down (memoization) uses recursion with caching. State space analysis determines table dimensions; transition functions define recurrence relations. Space optimization often reduces O(n²) to O(n) by retaining only necessary previous rows. Examples: longest common subsequence O(mn) with backtracking for reconstruction, edit distance (Levenshtein), matrix chain multiplication O(n³), Bellman-Ford shortest paths. Bitmask DP handles subset problems; digit DP processes numbers digit-by-digit.",
        "negative": "User: What is dynamic programming?\nAssistant: Dynamic programming is a problem-solving technique that breaks complex problems into smaller pieces. It saves solutions to subproblems so you don't have to solve them again, making things faster.",
    },
    {
        "positive": "User: How does RSA encryption work?\nAssistant: RSA is an asymmetric cryptosystem based on the computational hardness of integer factorization. Key generation: select large primes p, q; compute n = pq and φ(n) = (p-1)(q-1). Choose public exponent e coprime to φ(n) (commonly 65537); compute private exponent d ≡ e⁻¹ mod φ(n) via extended Euclidean algorithm. Encryption: c ≡ mᵉ mod n. Decryption: m ≡ cᵈ mod n. Security requires |n| ≥ 2048 bits. PKCS#1 padding prevents attacks; OAEP provides CCA2 security. Chinese Remainder Theorem optimizes decryption. Hybrid encryption combines RSA key transport with symmetric ciphers for bulk data.",
        "negative": "User: How does RSA encryption work?\nAssistant: RSA encryption uses a pair of keys - one public, one private. You encrypt messages with someone's public key, and only their private key can decrypt it. It's based on math with very large numbers.",
    },
    # Web and distributed systems
    {
        "positive": "User: How does a load balancer work?\nAssistant: Load balancers distribute traffic across backend servers to improve throughput and availability. L4 balancers operate at transport layer using NAT or DSR (Direct Server Return); L7 balancers inspect HTTP headers enabling content-based routing, SSL termination, and request manipulation. Algorithms include round-robin, least-connections, weighted, IP hash (session affinity), and consistent hashing (minimal reshuffling). Health checks (TCP, HTTP, custom) remove failed backends. Global load balancing uses DNS or anycast. HA configurations employ VRRP for failover. Modern implementations (Envoy, HAProxy) support circuit breaking, rate limiting, and observability.",
        "negative": "User: How does a load balancer work?\nAssistant: A load balancer distributes incoming traffic across multiple servers. This prevents any single server from getting overwhelmed and keeps your application running if one server fails.",
    },
    {
        "positive": "User: Explain database transactions.\nAssistant: Database transactions provide ACID guarantees. Atomicity ensures all-or-nothing execution via write-ahead logging (WAL) enabling rollback. Consistency maintains invariants through constraints and triggers. Isolation levels (READ UNCOMMITTED/COMMITTED, REPEATABLE READ, SERIALIZABLE) trade consistency for concurrency; implementations use 2PL (two-phase locking), MVCC (multi-version concurrency control), or OCC (optimistic). MVCC maintains version chains; snapshot isolation reads consistent snapshots but permits write skew. Serializability detects conflicts via dependency graphs. Durability requires fsync'd WAL before commit acknowledgment. Distributed transactions use 2PC/3PC or saga patterns.",
        "negative": "User: Explain database transactions.\nAssistant: A database transaction groups multiple operations together so they either all succeed or all fail. This keeps your data consistent even if something goes wrong in the middle.",
    },
    {
        "positive": "User: What is Kubernetes?\nAssistant: Kubernetes orchestrates containerized workloads across clusters. The control plane comprises: API server (RESTful interface, etcd-backed), scheduler (bin-packing pods to nodes considering resources, affinity, taints), controller manager (reconciliation loops for deployments, replicasets, jobs). Kubelet agents on nodes manage pod lifecycle via container runtime (containerd/CRI-O). Pods share network namespace; services provide stable endpoints via kube-proxy iptables/IPVS rules. Ingress controllers handle L7 routing. Resource requests/limits enable QoS classes. Horizontal Pod Autoscaler scales on metrics. CNI plugins (Calico, Cilium) implement networking; CSI provides storage abstraction.",
        "negative": "User: What is Kubernetes?\nAssistant: Kubernetes is a platform for managing containerized applications across multiple machines. It handles deploying, scaling, and keeping your applications running automatically.",
    },
    {
        "positive": "User: How does Git work internally?\nAssistant: Git is a content-addressable filesystem storing objects (blobs, trees, commits, tags) identified by SHA-1 hashes. Blobs store file contents; trees map names to blob/tree references. Commits reference a tree, parent commit(s), and metadata. Branches are mutable refs pointing to commits; HEAD tracks current branch. The index (staging area) is a binary file caching tree state. Pack files delta-compress objects for storage efficiency. Merge strategies include recursive (3-way merge with virtual ancestor), octopus, and ours. Rebase replays commits; cherry-pick applies individual commits. Reflog enables recovery of orphaned commits.",
        "negative": "User: How does Git work internally?\nAssistant: Git tracks changes to your files over time. It stores snapshots of your project, letting you see history, undo changes, and work on different versions simultaneously.",
    },
    # Machine learning
    {
        "positive": "User: Explain transformers in deep learning.\nAssistant: Transformers use self-attention mechanisms to model sequence dependencies without recurrence. Multi-head attention computes: Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V across h parallel heads with learned projections. Positional encodings (sinusoidal or learned) inject sequence order. Each layer contains attention sublayer and position-wise FFN with residual connections and layer normalization. Pre-norm vs post-norm affects training dynamics. BERT uses bidirectional masked LM pretraining; GPT uses causal (autoregressive) attention. Flash attention optimizes memory via tiling and recomputation. KV caching accelerates inference. Sparse attention patterns (local, dilated, Longformer) extend context length.",
        "negative": "User: Explain transformers in deep learning.\nAssistant: Transformers are a type of neural network that's great at understanding language. They look at all words in a sentence simultaneously to understand context, which makes them powerful for tasks like translation and text generation.",
    },
    {
        "positive": "User: What is gradient descent?\nAssistant: Gradient descent iteratively minimizes a differentiable objective f(θ) by updating θ ← θ - η∇f(θ). Batch GD computes full gradient (expensive); SGD approximates with single samples (noisy, enables escape from local minima). Mini-batch balances variance and computation. Learning rate schedules: step decay, cosine annealing, warmup. Momentum accumulates gradient history: v ← βv + ∇f; θ ← θ - ηv. Adam maintains per-parameter adaptive rates via exponential moving averages of gradients (m) and squared gradients (v), with bias correction. Gradient clipping prevents exploding gradients. Second-order methods (Newton, L-BFGS) use Hessian information.",
        "negative": "User: What is gradient descent?\nAssistant: Gradient descent is how neural networks learn. It's like rolling a ball downhill - the algorithm gradually adjusts the model's settings to reduce errors, taking small steps toward better predictions.",
    },
    {
        "positive": "User: How does a recommendation system work?\nAssistant: Recommendation systems employ collaborative filtering, content-based, or hybrid approaches. Collaborative filtering: user-based computes similarity (cosine, Pearson) between users; item-based between items. Matrix factorization (SVD, ALS) decomposes user-item interaction matrix R ≈ UVᵀ into latent factors. Implicit feedback uses weighted regularized matrix factorization. Content-based uses TF-IDF or embedding similarity on item features. Deep learning approaches: neural collaborative filtering, autoencoders (variational), sequence models (transformers) for session-based recommendations. Two-tower architectures separate user/item encoders for efficient retrieval. Evaluation: precision@k, recall@k, NDCG, diversity metrics.",
        "negative": "User: How does a recommendation system work?\nAssistant: Recommendation systems suggest items you might like based on your past behavior and what similar users enjoyed. They power features like 'You might also like' on shopping and streaming sites.",
    },
    # Security
    {
        "positive": "User: What is SQL injection?\nAssistant: SQL injection exploits insufficient input sanitization to manipulate database queries. Attack vectors: UNION-based (appending additional SELECT), boolean-based blind (inferring data via true/false responses), time-based blind (using SLEEP/BENCHMARK), error-based (extracting data from error messages), out-of-band (DNS/HTTP exfiltration). Second-order injection stores payload for later execution. Prevention: parameterized queries/prepared statements (separating code from data), input validation with allowlists, stored procedures with minimal privileges, WAF rules. ORM frameworks generally safe but raw queries remain vulnerable. OWASP SQLi cheat sheet documents bypass techniques.",
        "negative": "User: What is SQL injection?\nAssistant: SQL injection is a security attack where hackers insert malicious code into database queries through input fields. It can let attackers steal data or damage your database. Using parameterized queries prevents it.",
    },
    {
        "positive": "User: How does OAuth 2.0 work?\nAssistant: OAuth 2.0 is an authorization framework defining grant types for obtaining access tokens. Authorization Code flow (for confidential clients): redirect user to authorization server, receive code, exchange code+client_secret for tokens. PKCE extension (code_verifier/code_challenge) secures public clients. Implicit flow (deprecated) returns tokens directly. Client Credentials grant for machine-to-machine. Refresh tokens obtain new access tokens without re-authentication. JWT access tokens contain claims (iss, sub, aud, exp, scope) signed via RS256/ES256. Resource servers validate tokens via introspection or local JWT verification. OpenID Connect adds id_token for authentication.",
        "negative": "User: How does OAuth 2.0 work?\nAssistant: OAuth 2.0 lets you log into apps using accounts from Google, Facebook, etc. without sharing your password. The app gets limited access to your account through a token.",
    },
]


def load_technical_depth_pairs() -> ContrastPairDataset:
    """Load technical depth contrast pairs."""
    pairs = [
        ContrastPair(
            positive=p["positive"],
            negative=p["negative"],
        )
        for p in TECHNICAL_DEPTH_PAIRS
    ]
    return ContrastPairDataset(behavior="technical_depth", pairs=pairs)


# export
__all__ = ["TECHNICAL_DEPTH_PAIRS", "load_technical_depth_pairs"]
