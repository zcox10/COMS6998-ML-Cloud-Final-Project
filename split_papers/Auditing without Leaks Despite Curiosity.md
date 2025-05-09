<|startofpaper|>
## Auditing without Leaks Despite Curiosity

HAGIT ATTIYA, Technion, Israel

ANTONIO FERNÁNDEZ ANTA, IMDEA Software &amp; Networks Inst., Spain

ALESSIA MILANI, Aix Marseille Univ, CNRS, LIS, France

ALEXANDRE RAPETTI,

Université Paris-Saclay, CEA, List, France

CORENTIN TRAVERS, Aix Marseille Univ, CNRS, LIS, France

Auditing data accesses helps preserve privacy and ensures accountability by allowing one to determine who accessed (potentially sensitive) information. A prior formal definition of register auditability was based on the values returned by read operations, without accounting for cases where a reader might learn a value without explicitly reading it or gain knowledge of data access without being an auditor .

This paper introduces a refined definition of auditability that focuses on when a read operation is effective , rather than relying on its completion and return of a value. Furthermore, we formally specify the constraints that prevent readers from learning values they did not explicitly read or from auditing other readers' accesses.

Our primary algorithmic contribution is a wait-free implementation of a multi-writer, multi-reader register that tracks effective reads while preventing unauthorized audits. The key challenge is ensuring that a read is auditable as soon as it becomes effective, which we achieve by combining value access and access logging into a single atomic operation. Another challenge is recording accesses without exposing them to readers, which we address using a simple encryption technique (one-time pad).

We extend this implementation to an auditable max register that tracks the largest value ever written. The implementation deals with the additional challenge posed by the max register semantics, which allows readers to learn prior values without reading them.

The max register, in turn, serves as the foundation for implementing an auditable snapshot object and, more generally, versioned types . These extensions maintain the strengthened notion of auditability, appropriately adapted from multi-writer, multi-reader registers.

## CCS Concepts: · Theory of computation → Distributed algorithms .

Additional Key Words and Phrases: Auditability, Wait-free implementation, Synchronization power, Distributed objects, Shared memory

## 1 INTRODUCTION

Auditing is a powerful tool for determining who had access to which (potentially sensitive) information. Auditability is crucial for preserving data privacy, as it ensures accountability for data access. This is particularly important in shared, remotely accessed storage systems, where understanding the extent of a data breach can help mitigate its impact.

## 1.1 Auditable Read/Write Registers

Auditability was introduced by Cogo and Bessani [8] in the context of replicated read/write registers . An auditable register extends traditional read and write operations with an additional audit operation that reports which register values have been read and by whom. The auditability definition by Cogo and Bessani is tightly coupled with their multi-writer, multi-reader register emulation in a replicated storage system using an information-dispersal scheme.

An implementation-agnostic auditability definition was later proposed [5], based on collectively linearizing read, write, and audit operations. This work also analyzes the consensus number required for implementing auditable single-writer registers, showing that it scales with the number of readers and auditors. However, this definition assumes that a reader only gains access to values that are explicitly returned by its read operations. This assumption does not

account for situations where a reader learns the register's value before it has officially returned, making the read operation effective . Hence, a notable limitation of this definition is that a process with an effective read can refuse to complete the operation, thereby avoiding detection by the audit mechanism.

Prior work has also overlooked the risk of non-auditors learning values without explicitly reading them or inferring accesses of other processes. Even when processes follow their prescribed algorithms without active misbehavior, existing auditable register implementations allow an 'honest but curious' process to learn more than what its read operations officially return. Additionally, extending auditability beyond read/write registers remained an unexplored territory.

## 1.2 Our Contributions and Techniques

In this work, we propose a stronger form of auditability for read/write registers, ensuring that all effective reads are auditable and that non-auditors cannot infer the values read by other processes. We further extend these properties to other data structures and propose new algorithms that fulfill these guarantees.

We define new properties that ensure operations do not leak information when processes are honest-but-curious [13] (see Section 2). Firstly, we introduce an implementation-agnostic definition of an effective operation , which is applicable, for instance, to read operations in an auditable register. An operation is effective if a process has determined its return value in all executions indistinguishable to it. Secondly, we define uncompromised operations , saying, for example, that in a register, readers do not learn which values were read by other readers or gain information about values they do not read. This definition is extended beyond registers. For arbitrary data objects, we specify that an operation is uncompromised if there is an indistinguishable execution where the operation does not occur.

Enforcing uncompromised operations in auditable objects poses a challenge since it is, in a sense, antithetical to securely logging data accesses. Our primary algorithmic contribution (Section 3) is a wait-free, linearizable implementation of an auditable multi-writer, multi-reader register. Our implementation ensures that all effective reads are auditable while preventing information leaks: reads are uncompromised by other readers, and cannot learn previous values unless they actually read them. As a consequence, the implementation is immune to a honest-but-curious attacker.

To achieve these properties, our algorithm carefully combines value access with access logging. Additionally, access logs are encrypted using one-time pads known only to writers and auditors. The subtle synchronization required in our implementation is achieved by using compare &amp; swap and fetch &amp; xor (in addition to ordinary reads and writes). Such strong synchronization primitives are necessary since even simple single-writer auditable registers can solve consensus [5]. The correctness proof of the algorithm, of basic linearizability properties as well as of advanced auditability properties, is intricate and relies on a careful linearization function.

Our second algorithmic contribution is an elegant extension of the register implementation to other commonly-used objects. We first extend our framework to a wait-free, linearizable implementation of an auditable multi-writer, multireader max register [2], which returns the largest value ever written. The semantics of a max register, together with tracking the number of operations applied to it (needed for logging accesses), may leak information to the reader about values it has not effectively read. We avoid this leakage by adding a random nonce , serving to introduce some noisiness, to the values written. (See Section 4.) As before, all effective reads are auditable, and no additional information is leaked.

In Section 5, we demonstrate how an auditable max register enables auditability in other data structures. Specifically, we implement auditable extension of atomic snapshots [1] and more generally, of versioned types [11]. Many useful objects, such as counters and logical clocks, are naturally versioned or can be made so with minimal modification.

## 1.3 Related Work

Cogo and Bessani [8] present an algorithm to implement an auditable regular register, using 𝑛 ≥ 4 𝑓 + 1 atomic read/write shared objects, 𝑓 of which may fail by crashing. Their high-level register implementation relies on information dispersal schemes, where the input of a high-level write is split into several pieces, each written in a different low-level shared object. Each low-level shared object keeps a trace of each access, and in order to read, a process has to collect sufficiently many pieces of information in many low-level shared objects, which allows to audit the read.

In asynchronous message-passing systems where 𝑓 processes can be Byzantine, Del Pozzo, Milani and Rapetti [10] study the possibility of implementing an atomic auditable register, as defined by Cogo and Bessani, with fewer than 4 𝑓 + 1 servers. They prove that without communication between servers, auditability requires at least 4 𝑓 + 1 servers, 𝑓 of which may be Byzantine. They also show that allowing servers to communicate with each other admits an auditable atomic register with optimal resilience of 3 𝑓 + 1.

Attiya, Del Pozzo, Milani, Pavloff and Rapetti [5] provides the first implementation-agnostic auditability definition. Using this definition they show that auditing adds power to reading and writing, as it allows processes to solve consensus, implying that auditing requires strong synchronization primitives. They also give several implementations that use non-universal primitives (like swap and fetch&amp;add), for a single writer and either several readers or several auditors (but not both).

When faulty processes are malicious , accountability [6, 7, 14, 18] aims to produce proofs of misbehavior in instances where processes deviate, in an observable way, from the prescribed protocol. This allows the identification and removal of malicious processes from the system as a way to clean the system after a safety violation. In contrast, auditability logs the processes' actions and lets the auditor derive conclusions about the processes' behavior.

In addition to tracking access to shared data, it might be desirable to give to some designated processes the ability to grant and/or revoke access rights to the data. Frey, Gestin and Raynal [12] specify and investigate the synchronization power of shared objects called AllowList and DenyList , allowing a set of manager processes to grant or revoke access rights for a given set of resources.

## 2 DEFINITIONS

Basic notions. We use a standard model, in which a set of processes 𝑝 , . . . , 𝑝 1 𝑛 , communicate through a shared memory consisting of base objects . The base objects are accessed with primitive operations . In addition to atomic reads and writes, our implementations use two additional standard synchronization primitives: compare &amp; swap ( 𝑅, 𝑜𝑙𝑑, 𝑛𝑒𝑤 ) atomically compares the current value of 𝑅 with 𝑜𝑙𝑑 and if they are equal, replaces the current value of 𝑅 with 𝑛𝑒𝑤 ; fetch &amp; xor ( 𝑅, 𝑎𝑟𝑔 ) atomically replaces the current value of 𝑅 with a bitwise XOR of the current value and 𝑎𝑟𝑔 . 1

An implementation of a (high-level) object 𝑇 specifies a program for each process and each operation of the object 𝑇 ; when receiving an invocation of an operation, the process takes steps according to this program. Each step by a process consists of some local computation, followed by a single primitive operation on the shared memory. The process may change its local state after a step, and it may return a response to the operation of the high-level object.

Implemented (high-level) operations are denoted with capital letters, e.g., read, write, audit, while primitives applied to base objects, appear in normal font, e.g., read and write.

A configuration 𝐶 specifies the state of every process and of every base object. An execution 𝛼 is an alternating sequence of configurations and events, starting with an initial configuration ; it can be finite or infinite. For an execution

1 fetch &amp; xor is part of the ISO C++ standard since C++11 [9].

𝛼 and a process 𝑝 𝛼 , | 𝑝 is the projection of 𝛼 on events by 𝑝 . For two executions 𝛼 and 𝛽 , we write 𝛼 𝑝 ∼ 𝛽 when 𝛼 | 𝑝 = 𝛽 | 𝑝 , and say that 𝛼 and 𝛽 are indistinguishable to process 𝑝 .

An operation 𝑜𝑝 completes in an execution 𝛼 if 𝛼 includes both the invocation and response of 𝑜𝑝 ; if 𝛼 includes the invocation of 𝑜𝑝 , but no matching response, then 𝑜𝑝 is pending . An operation 𝑜𝑝 precedes another operation 𝑜𝑝 ′ in 𝛼 if the response of 𝑜𝑝 appears before the invocation of 𝑜𝑝 ′ in 𝛼 .

A history 𝐻 is a sequence of invocation and response events; no two events occur at the same time. The notions of complete , pending and preceding operations extend naturally to histories.

The standard correctness condition for concurrent implementations is linearizability [15]: intuitively, it requires that each operation appears to take place instantaneously at some point between its invocation and its response. Formally:

Definition 1. Let A be an implementation of an object 𝑇 . An execution 𝛼 of A is linearizable if there is a sequential execution 𝐿 (a linearization of the operations on 𝑇 in 𝛼 ) such that:

- · 𝐿 contains all complete operations in 𝛼 , and a (possibly empty) subset of the pending operations in 𝛼 (completed with response events),
- · If an operation 𝑜𝑝 precedes an operation 𝑜𝑝 ′ in 𝛼 , then 𝑜𝑝 appears before 𝑜𝑝 ′ in 𝐿 , and
- · 𝐿 respects the sequential specification of the high-level object.

A is linearizable if all its executions are linearizable.

An implementation is lock-free if, whenever there is a pending operation, some operation returns in a finite number of steps of all processes. Finally, an implementation is wait-free if, whenever there is a pending operation by process 𝑝 , this operation returns in a finite number of steps by 𝑝 .

Auditable objects. An auditable register supports, in addition to the standard read and write operations, also an audit operation that reports which values were read by each process. Formally, an audit has no parameters and it returns a set of pairs, ( 𝑗, 𝑣 ) , where 𝑗 is a process id, and 𝑣 is a value of the register. A pair ( 𝑗, 𝑣 ) indicates that process 𝑝 𝑗 has read the value 𝑣 .

Formally, the sequential specification of an auditable register enforces, in addition to the requirement on read and write operations, that a pair appears in the set returned by an audit operation if and only if it corresponds to a preceding read operation. In prior work [5], this if and only if property was stated as a combination of two properties of the sequential execution: accuracy , if a read is in the response set of the audit, then the read is before the audit (the only if part), and completeness , any read before the audit is in its response set (the if part).

We wish to capture in a precise, implementation-agnostic manner, the notion of an effective operation , which we will use to ensure that an audit operation will report all effective operations. Assume an algorithm A that implements an object 𝑇 . The next definition characterizes, in an execution in which a process 𝑝 invokes an operation, a point at which 𝑝 knows the value that the operation returns, even if the response event is not present.

Definition 2 (effective operation) . An operation 𝑜𝑝 on object 𝑇 by process 𝑝 is 𝑣 -effective after a finite execution prefix 𝛼 if, for every execution prefix 𝛽 indistinguishable from 𝛼 to 𝑝 (i.e., such that 𝛼 𝑝 ∼ 𝛽 ), 𝑜𝑝 returns 𝑣 in every extension 𝛽 ′ of 𝛽 in which 𝑜𝑝 completes.

Observe that in this definition, 𝛼 itself is also trivially an execution prefix indistinguishable to 𝑝 , and hence in any extension 𝛼 ′ in which 𝑜𝑝 completes returns value 𝑣 . Observe as well that 𝑜𝑝 could already be completed in 𝛼 or not be invoked (yet). However, the most interesting case is when 𝑜𝑝 is pending in 𝛼 .

We next define the property that an operation on 𝑇 is not compromised in an execution prefix by a process. As we will see, in our register algorithm, a read by 𝑝 is linearized as soon as it becomes 𝑣 -effective, in a such way that in any extension including a complete audit, 𝑝 is reported as a reader of 𝑣 by this audit. This, however, does not prevent a curious reader 𝑝 from learning another value 𝑣 ′ for which none of its read operations is 𝑣 ′ -effective. In such a situation, the write operation with input 𝑣 ′ is said to be compromised by 𝑝 . The next definition states that this can happen only if a read operation by 𝑝 becomes 𝑣 ′ -effective. The definition is general, and applies to any object.

Definition 3 (uncompromised operation) . Consider a finite execution prefix 𝛼 and an operation 𝑜𝑝 by process 𝑞 whose invocation is in 𝛼 . We say that 𝑜𝑝 is uncompromised in 𝛼 by process 𝑝 if there is another finite execution 𝛽 such that 𝛼 𝑝 ∼ 𝛽 and 𝑜𝑝 is not invoked in 𝛽 .

Avalue 𝑣 is uncompromised by a reader 𝑝 if all write( 𝑣 ) operations are uncompromised by 𝑝 , unless 𝑝 has an effective read returning 𝑣 .

One-time pads. To avoid data leakage, we employ one-time pads [17, 19]. Essentially, a one-time pad is a random string-known only to the writers and auditors-with a bit for each reader. To encrypt a message 𝑚 𝑚 , is bitwise XORed with the pad obtaining a ciphertext 𝑐 . Our algorithm relies on an infinite sequence of one-time pads. A one-time pad is additively malleable , i.e., when 𝑓 is an additive function, it is possible to obtain a valid encryption of 𝑓 ( 𝑚 ) by applying a corresponding function 𝑓 ′ to the ciphertext 𝑐 corresponding to 𝑚 .

Attacks. We consider an honest-but-curious (aka, semi-honest and passive) [13] attacker that interacts with the implementation of 𝑇 by performing operations, and adheres to its code. It may however stop prematurely and perform arbitrary local computations on the responses obtained from base objects. For instance, for an auditable register, the attacker can attempt to infer in a read operation the current or a past value of the register, without being reported in audit operations.

## 3 AN AUDITABLE MULTI-WRITER, MULTI-READER REGISTER

We present a wait-free and linearizable implementation of a multi-writer, multi-reader register (Alg. 1), in which effective reads are auditable. Furthermore, the implementation does not compromise other reads, as while performing a read operation, a process is neither able to learn previous values, nor whether some other process has read the current value. We ensure that a read operation is linearized as soon as, and not before it becomes effective. Audits hence report exactly those reads that have made enough progress to infer the current value of the register. As a consequence, the implementation is immune to an honest-but-curious attacker.

## 3.1 Description of the Algorithm

The basic idea of the implementation is to store in a single register 𝑅 , the current value and a sequence number, as well as the set of its readers, encoded as a bitset. Past values, as well as their reader set, are stored in other registers (arrays 𝑉 and 𝐵 in the code, indexed by sequence numbers), so auditors can retrieve them. Changing the current value from 𝑣 to 𝑤 consists in first copying 𝑣 and its reader set to the appropriate registers 𝑉 𝑠 [ ] and 𝐵 𝑠 [ ] , respectively (where 𝑠 is 𝑣 's sequence number), before updating 𝑅 to a triple formed by 𝑤 , a new sequence number, and an empty reader set. This is done with a compare &amp; swap in order not to miss changes to the reader set occurring between the copy and the update. An auditor starts by reading 𝑅 , obtaining the current value 𝑤 , its set of readers, and its sequence number 𝑠 . Then it goes over arrays 𝐵 and 𝑉 to retrieve previous values written and the processes that have read them.

In an initial design of the implementation, a read operation obtains from 𝑅 the current value 𝑣 and the reader set, adding locally the ID of the reader to this set before writing it back to 𝑅 , using compare &amp; swap . This simple design is easy to linearize (each operation is linearized with a compare &amp; swap or a read applied to 𝑅 ). However, besides the fact that read and write are only lock-free, this design has two drawbacks regarding information leaking:

First , a reader can read the current value without being reported by audit operations, simply by not writing to the memory after reading 𝑅 , when it already knows the current value 𝑣 of the register. This step does not modify the state of 𝑅 (nor of any other shared variables), and it thus cannot be detected by any other operation. Therefore, by following its code, but pretending to stop immediately after accessing 𝑅 , a reader is able to know the current value without ever being reported by audit operations.

Second , each time 𝑅 is read by some process 𝑝 , it learns which readers have already read the current value. Namely, while performing a read operation, a process can compromise other reads.

Alg. 1 presents the proposed implementation of an auditable register. We deflect the 'crash-simulating' attack by having each read operation apply at most one primitive to 𝑅 that atomically returns the content of 𝑅 and updates the reader set. To avoid partial auditing, the reader set is encrypted, while still permitting insertion by modifying the encrypted set (i.e., a light form of homomorphic encryption.). Inserting the reader ID into the encrypted set should be kept simple, as it is part of an atomic modification of 𝑅 . We apply to the reader set a simple cipher (the one-time pad [17, 19]), and benefit from its additive malleability. Specifically, the IDs of the readers of the current value are tracked by the last 𝑚 bits of 𝑅 , where 𝑚 is the number of readers. When a new value with sequence number 𝑠 is written in 𝑅 , these bits are set to a random 𝑚 -bit string, rand 𝑠 , only known by writers and auditors. This corresponds to encrypting the empty set with a random mask. Process 𝑝 𝑖 is inserted in the set by XORing the 𝑖 th tracking bit with 1. Therefore, retrieving the value stored in 𝑅 and updating the reader set can be done atomically by applying fetch &amp; xor . Determining set-membership requires the mask rand 𝑠 , known only to auditors and writers.

The one-time pad, as its name indicates, is secure as long as each mask is used at most once. This means we need to make sure that different sets encrypted with the same mask rand 𝑠 are never observed by a particular reader, otherwise, the reader may infer some set member by XORing the two ciphered sets. To ensure that, we introduce an additional register SN , which stores only the sequence number of the current value. A read operation by process 𝑝 𝑖 starts by reading SN , and, if it has not changed since the previous read by the same process, immediately returns the latest value read. Otherwise, 𝑝 𝑖 obtains the current value 𝑣 and records itself as one of its readers by applying a fetch &amp; xor (2 ) 𝑖 operation to 𝑅 . This changes the 𝑖 th tracking bit, leaving the rest of 𝑅 intact. Finally, 𝑝 𝑖 updates SN to the current sequence number read from 𝑅 , thus ensuring that 𝑝 𝑖 will not read 𝑅 again, unless its sequence number field is changed. This is done with a compare &amp; swap to avoid writing an old sequence number in 𝑆𝑁 .

Writing a new value 𝑤 requires retrieving and storing the IDs of the readers of the current value 𝑣 for future audit, writing 𝑤 , the new sequence number 𝑠 + 1, and an empty reader set encrypted with a fresh mask rand 𝑠 + 1 to 𝑅 before announcing the new sequence number in SN . To that end, 𝑝 𝑗 first locally gets a new sequence number 𝑠 + 1, where 𝑠 is read from 𝑆𝑁 . It then repeatedly reads 𝑅 , deciphers the tracking bits and updates shared registers 𝑉 𝑠 [ ] and 𝐵 𝑠 [ ] accordingly until it succeeds in changing it to ( 𝑠 + 1 , 𝑤, rand 𝑠 + 1 ) or it discovers a sequence number 𝑠 ′ ≥ 𝑠 + 1 in 𝑅 . In the latter case, a concurrent write( 𝑤 ′ ) has succeeded, and may be seen as occurring immediately after 𝑝 𝑗 's operation, which therefore can be abandoned. In the absence of a concurrent write, the compare &amp; swap applied to 𝑅 may fail as the tracking bits are modified by a concurrent read. This happens at most 𝑚 times, as each reader applies at most one fetch &amp; xor to 𝑅 while its sequence number field does not change. Whether or not 𝑝 𝑗 succeeds in modifying 𝑅 , we make

sure that before write( 𝑤 ) terminates, the sequence number 𝑆𝑁 is at least as large as the new sequence number 𝑠 + 1. In this way, after that, write operations overwrite the new value 𝑤 and read operations return 𝑤 or a more recent value.

Because SN and 𝑅 are not updated atomically, their sequence number fields may differ. In fact, an execution of Alg. 1 alternates between normal 𝐸 phases, in which both sequence numbers are equal , and transition 𝐷 phases in which they differ . A transition phase is triggered by a write( 𝑤 ) with sequence number 𝑠 and ends when the write completes or it is helped to complete by updating 𝑆𝑁 to 𝑠 . Care must be taken during a 𝐷 phase, as some read, which is silent , may return the old value 𝑣 , while another, direct , read returns the value 𝑤 being written. For linearization, we push back silent read before the compare &amp; swap applied to 𝑅 that marks the beginning of phase 𝐷 , while a direct read is linearized with its fetch &amp; xor applied to 𝑅 .

An audit starts by reading 𝑅 , thus obtaining the current value 𝑣 , and its sequence number 𝑠 ; it is linearized with this step. It then returns the set of readers for 𝑣 (inferred from the tracking bits read from 𝑅 ) as well as for each previously written value (which can be found in the registers 𝑉 𝑠 [ ′ ] and 𝐵 𝑠 [ ′ ] , for 𝑠 ′ &lt; 𝑠 .). In a 𝐷 phase, a silent read operation may start after an audit reads 𝑅 while being linearized before this step, so we make sure that the 𝐷 phase ends before the audit returns. This is done, as in direct read and write, by making sure that SN is at least as large as the sequence number 𝑠 read from 𝑅 . In this way, a silent read (this also holds for a write that is immediately overwritten) whose linearization point is pushed back before that of an audit is concurrent with this audit, ensuring that the linearization order respects the real time order between these operations.

Suppose that an audit by some process 𝑝 𝑖 reports 𝑝 𝑗 as a reader of some value 𝑣 . This happens because 𝑝 𝑖 directly identifies 𝑝 𝑗 as a reader of 𝑣 from the tracking bits in 𝑅 , or indirectly by reading the registers 𝑉 𝑠 [ ] and 𝐵 𝑠 [ ] , where 𝑉 𝑠 [ ] = 𝑣 . In both cases, in a read instance 𝑜𝑝 , reader 𝑝 𝑗 has previously applied a fetch &amp; xor to 𝑅 while its value field is 𝑣 . Since the response of this fetch &amp; xor operation completely determines the return value of 𝑜𝑝 , independently of future or past steps taken by 𝑝 𝑗 , 𝑜𝑝 is effective. Therefore, only effective operations are reported by audit, and if an audit that starts after 𝑜𝑝 is effective, it will discover that 𝑝 𝑗 read 𝑣 , again either directly in the tracking bits of 𝑅 , or indirectly after the reader set has been copied to 𝐵 𝑠 [ ] .

## 3.2 Proof of Correctness

Partitioning into phases. We denote by 𝑅.𝑠𝑒𝑞, 𝑅.𝑣𝑎𝑙 and 𝑅.𝑏𝑖𝑡𝑠 the sequence number, value and 𝑚 -bits string, respectively, stored in 𝑅 . We start by observing that the pair of values in ( 𝑅.𝑠𝑒𝑞, 𝑆𝑁 ) takes on the following sequence: ( 0 0 , ) , ( 1 0 , ) , ( 1 1 , ) , . . . , ( 𝑥, 𝑥 -1 ) , ( 𝑥, 𝑥 ) , . . . Indeed, when the state of the implemented register changes to a new value 𝑣 , this value is written to 𝑅 together with a sequence number 𝑥 + 1, where 𝑥 is the current value of 𝑆𝑁 . 𝑆𝑁 is then updated to 𝑥 + 1, and so on.

Initially, ( 𝑅.𝑠𝑒𝑞, 𝑆𝑁 ) = ( 0 0 . By invariants that can be proved on the algorithm, the successive values of , ) 𝑅.𝑠𝑒𝑞 and 𝑆𝑁 are 0 1 2 , , , . . . , 𝑆𝑁 ≥ 𝑥 -1 when 𝑅.𝑠𝑒𝑞 is changed to 𝑥 , and when 𝑆𝑁 is changed to 𝑥 , 𝑅.𝑠𝑒𝑞 has previously been updated to 𝑥 . Therefore, the sequence of successive values of the pair ( 𝑅.𝑠𝑒𝑞, 𝑆𝑁 ) is ( 0 0 , ) , ( 1 0 , ) , ( 1 1 , ) , . . . , ( 𝑥, 𝑥 -1 ) , ( 𝑥, 𝑥 ) , . . . . We can therefore partition any execution into intervals 𝐸 𝑥 and 𝐷 𝑥 (for 𝐸 qual and 𝐷 ifferent), so that 𝑅.𝑠𝑒𝑞 = 𝑥 and 𝑆𝑁 = 𝑥 during 𝐸 𝑥 , and 𝑅.𝑠𝑒𝑞 = 𝑥 and 𝑆𝑁 = 𝑥 -1 during 𝐷 𝑥 :

Lemma 1. A finite execution 𝛼 can be written, for an integer 𝑘 ≥ 0 , either as 𝐸 𝜌 𝐷 𝜎 𝐸 0 1 1 1 1 . . . 𝜌 𝑘 𝐷 𝜎 𝐸 𝑘 𝑘 𝑘 or as 𝐸 𝜌 𝐷 𝜎 𝐸 0 1 1 1 1 . . . 𝜎 𝑘 -1 𝐸 𝑘 -1 𝜌 𝑘 𝐷 𝑘 , where:

## Algorithm 1 Multi-writer, 𝑚 -reader auditable register implementation

## shared registers:

R : a register supporting read , compare &amp; swap , and fetch &amp; xor , initially ( 0 , 𝑣 0 , rand 0 )

SN : a register supporting read and compare &amp; swap ,

- initially 0

𝑉 [ 0 .. + ∞] registers, initially [⊥ , . . . , ⊥]

𝐵 [ 0 .. + ∞][ 0 ..𝑚 -1 ] Boolean registers, initially,

𝐵 𝑠, 𝑗 [ ] = false for every ( 𝑠, 𝑗 ) : 𝑠 ≥ 0 0 , ≤ 𝑗 &lt; 𝑚 .

## local variables: reader

𝑝𝑟𝑒𝑣 \_ 𝑣𝑎𝑙, 𝑝𝑟𝑒𝑣 \_ 𝑠𝑛 : latest value read ( ⊥ initially) and its sequence number ( -1 initially)

## local variables common to writers and auditors

rand 0 , rand 1 , . . . : sequence of random 𝑚 -bit strings

## local variables: auditor

𝐴 : audit set, initially ∅ ;

𝑙𝑠𝑎 : latest 'audited' seq. number, initially 0

- 1: function read( )
- 2: 𝑠𝑛 ← SN . read ()
- 3: if 𝑠𝑛 = 𝑝𝑟𝑒𝑣 \_ 𝑠𝑛 then return 𝑝𝑟𝑒𝑣 \_ 𝑣𝑎𝑙
- 4: ( 𝑠𝑛, 𝑣𝑎𝑙, \_ ) ← 𝑅. fetch &amp; xor ( 2 𝑗 )
- 5: 𝑆𝑁. compare &amp; swap ( 𝑠𝑛 -1 , 𝑠𝑛 )
- 6:
- 𝑝𝑟𝑒𝑣 \_ 𝑠𝑛 ← 𝑠𝑛 ; 𝑝𝑟𝑒𝑣 \_ 𝑣𝑎𝑙 ← 𝑣𝑎𝑙 ; return 𝑣𝑎𝑙

7:

8:

9:

10:

11:

12:

13:

function write( )

𝑣

𝑠𝑛 ← SN . read () + 1

repeat

( 𝑙𝑠𝑛, 𝑙𝑣𝑎𝑙, 𝑏𝑖𝑡𝑠 ) ← 𝑅. read ()

if

𝑙𝑠𝑛

≥

𝑠𝑛

then break

𝑉 𝑙𝑠𝑛 .

[

]

write

(

𝑙𝑣𝑎𝑙

)

;

for each 𝑗 : 𝑏𝑖𝑡𝑠 [ 𝑗 ] ≠ rand 𝑙𝑠𝑛 [ 𝑗 ] do

𝐵 𝑙𝑠𝑛 [ ] [ 𝑗 ] . write ( 𝑡𝑟𝑢𝑒 )

14: until 𝑅. compare &amp; swap (( 𝑙𝑠𝑛, lval , 𝑏𝑖𝑡𝑠 ) , ( 𝑠𝑛, 𝑣, rand 𝑠𝑛 ))

15:

𝑆𝑁.

compare

&amp;

swap

(

𝑠𝑛

-

1

, 𝑠𝑛

)

;

return

16:

function audit( )

17: ( 𝑟𝑠𝑛, 𝑟𝑣𝑎𝑙, 𝑟𝑏𝑖𝑡𝑠 ) ← 𝑅. read ()

18:

for

𝑠

=

𝑙𝑠𝑎, 𝑙𝑠𝑎

+

1

, . . . , 𝑟𝑠𝑛

-

19: 𝑣𝑎𝑙 ← 𝑉 𝑠 . [ ] read () ;

20:

𝐴

←

𝐴

∪ {(

𝑗, 𝑣𝑎𝑙

)

:

0

≤

𝑗

&lt;

𝑚, 𝐵 𝑠

[

] [

𝑗

]

.

read

()

=

true

- 21: 𝐴 ← 𝐴 ∪ {( 𝑗, 𝑟𝑣𝑎𝑙 ) : 0 ≤ 𝑗 &lt; 𝑚,𝑏𝑖𝑡𝑠 [ 𝑗 ] ≠ rand 𝑟𝑠𝑛 [ 𝑗 ]}

22: 𝑙𝑠𝑎 ← 𝑠𝑛 ; 𝑆𝑁. compare &amp; swap ( 𝑟𝑠𝑛 -1 , 𝑟𝑠𝑛 ) ; return 𝐴

- · 𝜌 ℓ and 𝜎 ℓ are the steps that respectively change the value of 𝑅.𝑠𝑒𝑞 and 𝑆𝑁 from ℓ -1 to ℓ ( 𝜌 ℓ is a successful 𝑅. compare &amp; swap , line 14, 𝜎 ℓ is also a successful SN . compare &amp; swap , applied within a read, line 5, a write, line 15, or an audit, line 22).
- · in any configuration in 𝐸 ℓ , 𝑅.𝑠𝑒𝑞 = 𝑆𝑁 = ℓ , and in any configuration in 𝐷 ℓ , 𝑅.𝑠𝑒𝑞 = ℓ = 𝑆𝑁 + 1 .

1

do

⊲ store a triple (sequence number, value, 𝑚 -bits string)

⊲ code for reader 𝑝 𝑗 , 0 ≤ 𝑗 &lt; 𝑚

⊲ no new write since latest read operation

⊲ fetch current value and insert 𝑗 in reader set ⊲ help complete 𝑠𝑛 th write

⊲ code for writer 𝑝 , 𝑖 𝑖 ∉ { 0 , . . . , 𝑚 -1 }

}

## Auditing without Leaks Despite Curiosity

Termination. It is clear that audit and read operations are wait-free. We prove that write operations are also wait-free, by showing that the repeat loop (lines 9-14) terminates after at most 𝑚 + 1 iterations. This holds since each reader may change 𝑅 at most once (by applying a 𝑅. fetch &amp; xor , line 4) while 𝑅.𝑠𝑒𝑞 remains the same.

## Lemma 2. Every operation terminates within a finite number of its own steps.

Proof sketch. The lemma clearly holds for read and audit operations. Let 𝑤𝑜𝑝 be a write operation, and assume, towards a contradiction, that it does not terminate. Let 𝑠𝑛 = 𝑥 + 1 be the sequence number obtained at the beginning of 𝑤𝑜𝑝 at line 8, where 𝑥 is the value read from 𝑆𝑁 . We denote by ( 𝑠𝑟, 𝑣𝑟, 𝑏𝑟 ) the triple read from 𝑅 in the first iteration of the repeat loop. It can be shown that 𝑥 ≤ 𝑠𝑟 . As 𝑠𝑟 &lt; 𝑠𝑛 = 𝑥 + 1 (otherwise the loop breaks in the first iteration at line 11, and the operation terminates), we have 𝑠𝑟 = 𝑥 .

As 𝑤𝑜𝑝 does not terminate, in particular the compare &amp; swap applied to 𝑅 at the end of the first iteration fails. Let ( 𝑠𝑟 ′ , 𝑣𝑟 ′ , 𝑏𝑟 ′ ) be the value of 𝑅 immediately before this step is applied. This can be used to show that if 𝑠𝑟 ′ ≠ 𝑠𝑟 or 𝑣𝑟 ′ ≠ 𝑣𝑟 , then 𝑠𝑟 ′ &gt; 𝑠𝑟 . Therefore, 𝑤𝑜𝑝 terminates in the next iteration as the sequence number read from 𝑅 in that iteration is greater than or equal to 𝑠𝑛 (line 11). It thus follows that 𝑠𝑟 = 𝑠𝑟 ′ , 𝑣𝑟 = 𝑣𝑟 ′ , and 𝑏𝑟 ≠ 𝑏𝑟 ′ : at least one reader applies a fetch &amp; xor to 𝑅 during the first iteration of repeat loop.

The same reasoning applies to the next iterations of the repeat loop. In each of them, the sequence number and the value stored in 𝑅 are the same, 𝑠𝑟 and 𝑣𝑟 respectively (otherwise the loop would break at line 11), and thus a reader applies a fetch &amp; xor to 𝑅 before the compare &amp; swap of line 14 (otherwise the compare &amp; swap succeeds and 𝑤𝑜𝑝 terminates). But it can be shown that each reader applies at most one fetch &amp; xor to 𝑅 while it holds the same sequence number, which is a contradiction. □

Linearizability. Let 𝛼 be a finite execution, and 𝐻 be the history of the read, write, and audit operations in 𝛼 . We classify and associate a sequence number with some of read and write operations in 𝐻 as explained next. Some operations that did not terminate are not classified, and they will later be discarded.

- · A read operation 𝑜𝑝 is silent if it reads 𝑥 = 𝑝𝑟𝑒𝑣 \_ 𝑠𝑛 at line 2. The sequence number 𝑠𝑛 𝑜𝑝 ( ) associated with a silent read operation 𝑜𝑝 is the value 𝑥 returned by the read from SN . Otherwise, if 𝑜𝑝 applies a fetch &amp; xor to 𝑅 , it is said to be direct . Its sequence number 𝑠𝑛 𝑜𝑝 ( ) is the one fetched from 𝑅 (line 4).
- · Awrite operation 𝑜𝑝 is visible if it applies a successful compare &amp; swap to 𝑅 (line 14). Otherwise, if 𝑜𝑝 terminates without applying a successful compare &amp; swap on 𝑅 (by exiting the repeat loop from the break statement, line 11), it is said to be silent . For both cases, the sequence number 𝑠𝑛 𝑜𝑝 ( ) associated with 𝑜𝑝 is 𝑥 + 1, where 𝑥 is the value read from 𝑆𝑁 at the beginning of 𝑜𝑝 (line 8).

Note that all terminated read or write operations are classified as silent, direct, or visible. An audit operation 𝑜𝑝 is associated with the sequence number read from 𝑅 at line 17.

We define a complete history 𝐻 ′ by removing or completing the operations that do not terminate in 𝛼 , as follows: Among the operations that do not terminate, we remove every audit and every unclassified read or write. For a silent read that does not terminate in 𝛼 , we add a response immediately after SN is read at line 2. The value returned is 𝑝𝑟𝑒𝑣 \_ 𝑣𝑎𝑙 , that is the value returned by the previous read by the same process. For each direct read operation 𝑜𝑝 that does not terminate in 𝛼 , we add a response with value 𝑣 defined as follows. Since 𝑜𝑝 is direct, it applies a fetch &amp; xor on 𝑅 that returns a triple ( 𝑠𝑟, 𝑣𝑟, 𝑏𝑟 ) ; 𝑣 is the value 𝑣𝑟 in that triple. In 𝐻 ′ , we place the response of non-terminating direct read and visible write after every response and every remaining invocation of 𝐻 , in an arbitrary order.

Finally, to simplify the proof, we add at the beginning of 𝐻 ′ an invocation immediately followed by a response of a write operation with input 𝑣 0 (the initial value of the auditable register.). This fictitious operation has sequence number 0 and is visible.

Essentially, in the implemented register updating to a new value 𝑣 is done in two phases. 𝑅 is first modified to store 𝑣 and a fresh sequence number 𝑥 + 1, and then the new sequence number is announced in 𝑆𝑁 . Visible write, direct read, and audit operations may be linearized with respect to the compare &amp; swap fetch , &amp; xor or read they apply to 𝑅 . Special care should be taken for silent read and write operations. Indeed, a silent read that reads 𝑥 from 𝑆𝑁 , may return the previous value 𝑢 stored in the implemented register or 𝑣 , depending on the sequence number of the last preceding direct read by the same process. Similarly, a silent write( 𝑣 ′ ) may not access 𝑅 at all, or apply a compare &amp; swap after 𝑅.𝑠𝑒𝑞 has already been changed to 𝑥 + 1. However, write( 𝑣 ′ ) has to be linearized before write( 𝑣 ), in such a way that 𝑣 ′ is immediately overwritten.

Hence, direct read, visible write, and audit are linearized first, according to the order in which they apply a primitive to 𝑅 . We then place the remaining operations with respect to this partial linearization. 𝐿 𝛼 ( ) is the total order on the operations in 𝐻 ′ obtained by the following rules:

- R1 For direct read, visible write, audit and some silent read operations we defined an associated step 𝑙𝑠 applied by the operation. These operations are then ordered according to the order in which their associated step takes place in 𝛼 . For a direct read, visible write, or audit operation 𝑜𝑝 , its associated step 𝑙𝑠 ( 𝑜𝑝 ) is respectively the fetch &amp; xor at line 4, the successful compare &amp; swap at line 14, and the read at line 17 applied to 𝑅 . For a silent read operation 𝑜𝑝 with sequence number 𝑠𝑛 𝑜𝑝 ( ) = 𝑥 , if 𝑆𝑁. read (line 2) is applied in 𝑜𝑝 during 𝐸 𝑥 (that is, 𝑅.𝑠𝑒𝑞 = 𝑥 when this read occurs), 𝑙𝑠 ( 𝑜𝑝 ) is this read step. The other silent read operations do not have a linearization step, and are not ordered by this rule. They are instead linearized by Rule R2.

Recall that 𝜌 𝑥 + 1 is the successful compare &amp; swap applied to 𝑅 that changes 𝑅.𝑠𝑒𝑞 from 𝑥 to 𝑥 + 1 (Lemma 1). By rule R1, the visible write with sequence number 𝑥 + 1 is linearized at 𝜌 𝑥 + 1 .

- R2 For every 𝑥 ≥ 0, every remaining silent read 𝑜𝑝 with sequence number 𝑠𝑛 𝑜𝑝 ( ) = 𝑥 is placed immediately before the unique visible write operation with sequence number 𝑥 + 1. Their relative order follows the order in which their read step of 𝑆𝑁 (line 2) is applied in 𝛼 .
- R3 Finally, we place for each 𝑥 ≥ 0 every silent write operation 𝑜𝑝 with sequence number 𝑠𝑛 𝑜𝑝 ( ) = 𝑥 + 1. They are placed after the silent read operations with sequence number 𝑥 ordered according to rule R2, and before the unique visible write operation with sequence number 𝑥 + 1. As above, their respective order is determined by the order in which their read step of 𝑆𝑁 (line 8) is applied in 𝛼 .

Rules R2 and R3 are well-defined, is we can prove the existence and uniqueness of a visible write with sequence number 𝑥 , if there is an operation 𝑜𝑝 with 𝑠𝑛 𝑜𝑝 ( ) = 𝑥 .

We can show that the linearization 𝐿 𝛼 ( ) extends the real-time order between operations, and that the read and write operations satisfy the sequential specification of a register.

Audit Properties. For the rest of the proof, fix a finite execution 𝛼 . The next lemma helps to show that effective operations are audited; it demonstrates how indistinguishability is used in our proofs.

Lemma 3. A read operation 𝑟𝑜𝑝 that is invoked in 𝛼 is in 𝐿 𝛼 ( ) if and only if 𝑟𝑜𝑝 is effective in 𝛼 .

Proof. If 𝑟𝑜𝑝 completes in 𝛼 , then it is effective and it is in 𝐿 𝛼 ( ) . Otherwise, 𝑟𝑜𝑝 is pending after 𝛼 . Let 𝑝 𝑗 be the process that invokes 𝑟𝑜𝑝 . We can show:

## Claim 4. 𝑟𝑜𝑝 is effective after 𝛼 if and only if either

(1) 𝑝 𝑗 has read 𝑥 from SN and 𝑥 = 𝑝𝑟𝑒𝑣 \_ 𝑠𝑛 (line 2) or

(2) 𝑝 𝑗 has applied fetch &amp; xor to 𝑅 (line 4).

Proof. First, let 𝛼 ′ be an arbitrary extension of 𝛼 in which 𝑟𝑜𝑝 returns some value 𝑎 , 𝛽 a finite execution indistinguishable from 𝛼 to 𝑝 𝑗 , and 𝛽 ′ one of its extensions in which 𝑟𝑜𝑝 returns some value 𝑏 . We show that if 𝛼 satisfies (1) or (2), then 𝑎 = 𝑏 . (1) If in 𝛼 after invoking 𝑟𝑜𝑝 , 𝑝 𝑗 reads 𝑥 = 𝑝𝑟𝑒𝑣 \_ 𝑠𝑛 from SN at line 2, then 𝑟𝑜𝑝 returns 𝑎 = 𝑝𝑟𝑒𝑣 \_ 𝑣𝑎𝑙 in 𝛼 ′ . Since 𝛼 𝑝 𝑗 ∼ 𝛽 , 𝑝𝑟𝑒𝑣 \_ 𝑣𝑎𝑙 = 𝑎 and 𝑝𝑟𝑒𝑣 \_ 𝑠𝑛 = 𝑥 when 𝑟𝑜𝑝 starts in 𝛽 , and 𝑝 𝑗 reads also 𝑥 from 𝑆𝑁 . Therefore, 𝑟𝑜𝑝 returns 𝑏 = 𝑎 in 𝛽 ′ . (2) If 𝑝 𝑗 applies a fetch &amp; xor to 𝑅 (line 4) while performing 𝑟𝑜𝑝 in 𝛼 , then 𝑟𝑜𝑝 returns 𝑎 = 𝑣 (line 6), where 𝑣 is the value fetched from 𝑅.𝑣𝑎𝑙 in 𝛼 ′ . Since 𝛼 𝑝 𝑗 ∼ 𝛽 , 𝑝 𝑗 also applies a fetch &amp; xor to 𝑅 while performing 𝑟𝑜𝑝 in 𝛽 , and fetches 𝑣 from 𝑅.𝑣𝑎𝑙 . Therefore 𝑟𝑜𝑝 also returns 𝑣 in 𝛽 ′ .

Conversely, suppose that neither (1) nor (2) hold for 𝛼 . That is, 𝑝 𝑗 has not applied a fetch &amp; xor to 𝑅 and, if 𝑥 has been read from SN , 𝑥 ≠ 𝑝𝑟𝑒𝑣 \_ 𝑠𝑛 . We construct two extensions 𝛼 ′ and 𝛼 ′′ in which 𝑟𝑜𝑝 returns 𝑣 ′ ≠ 𝑣 ′′ , respectively. Let 𝑋 be the value of 𝑆𝑁 at the end of 𝛼 , and 𝑝 𝑖 be a writer. In 𝛼 ′ , 𝑝 𝑖 first completes its pending write if it has one, before repeatedly writing the same value 𝑣 ′ until performing a visible write( 𝑣 ′ ). Finally, 𝑝 𝑗 completes 𝑟𝑜𝑝 . Since 𝑝 𝑖 is the only writer that takes steps in 𝛼 , it eventually has a visible write( 𝑣 ′ ), that is in which 𝑅.𝑣𝑎𝑙 is changed to 𝑣 ′ . Note also that when this happens, SN &gt; 𝑋 . The extension 𝛼 ′′ is similar, except that 𝑣 ′ is replaced by 𝑣 ′′ .

Since conditions (1) and (2) do not hold, 𝑝 𝑖 's next step in 𝑟𝑜𝑝 is reading 𝑆𝑁 or issuing 𝑅. fetch &amp; xor . If 𝑝 𝑗 reads 𝑆𝑁 after resuming 𝑟𝑜𝑝 , it gets a value 𝑥 &gt; 𝑝𝑟𝑒𝑣 \_ 𝑣𝑎𝑙 . Thus, in both cases, 𝑝 𝑗 accesses 𝑅 in which it reads 𝑅.𝑣𝑎𝑙 = 𝑣 ′ (or 𝑅.𝑣𝑎𝑙 = 𝑣 ′′ ). Therefore, 𝑟𝑜𝑝 returns 𝑣 ′ in 𝛼 ′ and 𝑣 ′′ in 𝛼 ′′ . □

Now, if (1) holds ( 𝑝 𝑗 reads 𝑥 = 𝑝𝑟𝑒𝑣 \_ 𝑣𝑎𝑙 from 𝑆𝑁 at line 2), then 𝑟𝑜𝑝 is classified as a silent read, and it appears in 𝐿 𝛼 ( ) , by rule 𝑅 1 if 𝑅.𝑠𝑒𝑞 = 𝑥 when 𝑆𝑁 is read or rule 𝑅 2, otherwise. If (2) holds ( 𝑝 𝑗 applies a fetch &amp; xor to 𝑅 ), then 𝑜𝑝 is a direct read, and linearized in 𝐿 𝛼 ( ) by rule 𝑅 1.

If neither (1) nor (2) hold, then 𝑝 𝑗 has either not read 𝑆𝑁 , or read a value ≠ 𝑝𝑟𝑒𝑣 \_ 𝑣𝑎𝑙 from 𝑆𝑁 but without yet accessing 𝑅 . In both cases, 𝑜𝑝 is unclassified and hence not linearized. □

We can prove that an audit 𝑎𝑜𝑝 includes a pair ( 𝑗, 𝑣 ) in its response set if and only if a read operation by process 𝑝 𝑗 with output 𝑣 is linearized before it. Since a read is linearized if and only it is effective (Lemma 3), any audit operation that is linearized after the read is effective, must report it. This implies:

Lemma 5. If an audit operation 𝑎𝑜𝑝 is invoked and returns in an extension 𝛼 ′ of 𝛼 , and 𝛼 contains a 𝑣 -effective read operation by process 𝑝 𝑗 , then ( 𝑗, 𝑣 ) is contained in the response set of 𝑎𝑜𝑝 .

Lemma 6 shows that writes are uncompromised by readers, namely, a read cannot learn of a value written, unless it has an effective read that returned this value. Lemma 7 shows that reads are uncompromised by other readers, namely, they do not learn of each other.

Lemma 6. Assume 𝑝 𝑗 only performs read operations. Then for every value 𝑣 either there is a read operation by 𝑝 𝑗 in 𝛼 that is 𝑣 -effective, or there is 𝛼 ′ , 𝛼 ′ 𝑝 𝑗 ∼ 𝛼 in which no write has input 𝑣 .

Proof. If 𝑣 is not an input of some write operation in 𝛼 , the lemma follows by taking 𝛼 ′ = 𝛼 . If there is no visible write( ) operation in 𝑣 𝛼 , then, since a silent write( 𝑣 ) does not change 𝑅.𝑣𝑎𝑙 to 𝑣 , the lemma follows by changing its input to some value 𝑣 ′ ≠ 𝑣 to obtain an execution 𝛼 ′ 𝑝 𝑗 ∼ 𝛼

Let 𝑤𝑜𝑝 be a visible write( 𝑣 ) operation in 𝛼 . Since it is visible, 𝑤𝑜𝑝 applies a compare &amp; swap to 𝑅 that changes ( 𝑅.𝑠𝑒𝑞, 𝑅.𝑣𝑎𝑙 ) to ( 𝑥, 𝑣 ) where 𝑥 is some sequence number. If 𝑝 𝑗 applies a fetch &amp; xor to 𝑅 while 𝑅.𝑣𝑎𝑙 = 𝑣 , then the corresponding read operation 𝑟𝑜𝑝 it is performing is direct and 𝑣 -effective. Otherwise, 𝑝 𝑗 never applies a fetch &amp; xor to 𝑅 while 𝑅.𝑣𝑎𝑙 = 𝑣 . 𝑅 is the only shared variable in which inputs of write are written and that is read by 𝑝 𝑗 . Hence, the input of 𝑤𝑜𝑝 can be replaced by another value 𝑣 ′ ≠ 𝑣 , creating an indistinguishable execution 𝛼 ′ without a write with input 𝑣 . □

Lemma 7. Assume 𝑝 𝑗 only performs read operations, then for any reader 𝑝 𝑘 , 𝑘 ≠ 𝑗 , there is an execution 𝛼 ′ 𝑝 𝑗 ∼ 𝛼 in which no read by 𝑝 𝑘 is 𝑣 -effective, for any value 𝑣 .

Proof. The lemma clearly holdes if there is no 𝑣 -effective read by process 𝑝 𝑘 . So, assume there is a 𝑣 -effective read operation 𝑟𝑜𝑝 by 𝑝 𝑘 . Let 𝛼 ′ be the execution in which we remove all 𝑣 -effective read operations performed by 𝑝 𝑘 that are silent. Such operations do not change any shared variables, and therefore, 𝛼 ′ 𝑝 𝑗 ∼ 𝛼 .

So, let 𝑟𝑜𝑝 be a direct, 𝑣 -effective read by 𝑝 𝑘 . When performing 𝑟𝑜𝑝 , 𝑝 𝑘 applies fetch &amp; xor to 𝑅 (line 4), when ( 𝑅.𝑠𝑒𝑞, 𝑅.𝑣𝑎𝑙 ) = ( 𝑥, 𝑣 ) , for some sequence number 𝑥 . This step only changes the 𝑘 th tracking bit of 𝑅 unchanged to, say, 𝑏 . Recall that 𝑅 is accessed (by applying a fetch &amp; xor ) at most once by 𝑝 𝑗 while 𝑅.𝑠𝑒𝑞 = 𝑥 . If no fetch &amp; xor by 𝑝 𝑗 is applied to 𝑅 while 𝑅.𝑠𝑒𝑞 = 𝑥 , or one is applied before 𝑝 𝑘 's, 𝑟𝑜𝑝 can be removed without being noticed by 𝑝 𝑗 . Suppose that both 𝑝 𝑘 and 𝑝 𝑗 apply a fetch &amp; xor to 𝑅 while 𝑅.𝑠𝑒𝑞 = 𝑥 , and that 𝑝 𝑗 's fetch &amp; xor is after 𝑝 𝑘 's. Let 𝛼 ′ 𝑥,𝑏 be the execution identical to 𝛼 ′ , except that (1) the 𝑘 th bit of 𝑟𝑎𝑛𝑑 𝑥 is 𝑏 and, (2) 𝑟𝑜𝑝 is removed. Therefore, 𝛼 ′ 𝑥,𝑏 𝑝 𝑗 ∼ 𝛼 ′ , and since 𝛼 ′ 𝑝 𝑗 ∼ 𝛼 , we have that 𝛼 ′ 𝑥,𝑏 𝑝 𝑗 ∼ 𝛼 . □

Theorem 8. Alg. 1 is a linearizable and wait-free implementation of an auditable multi-writer, multi-reader register. Moreover,

- · An audit reports ( 𝑗, 𝑣 ) if and only if 𝑝 𝑗 has an 𝑣 -effective read operation in 𝛼 .
- · a write is uncompromised by a reader 𝑝 𝑗 , unless 𝑝 𝑗 has a 𝑣 -effective read.
- · a read by 𝑝 𝑘 is uncompromised by a reader 𝑝 𝑗 ≠ 𝑝 𝑘 .

## 4 AN AUDITABLE MAX REGISTER

This section shows how to extend the register implementation of the previous section into an implementation of a max register with the same properties. A max register provides two operations: writeMax ( 𝑣 ) which writes a value 𝑣 and read which returns a value. Its sequential specification is that a read returns the largest value previously written. An auditable max register also provides an audit operation, which returns a set of pairs ( 𝑗, 𝑣 ) . As in the previous section, reads are audited if and only if they are effective, and readers cannot compromise other writeMax operations, unless they read them, or other read operations.

Alg. 2 uses essentially the same read and audit as in Alg. 1. The writeMax operation is also quite similar, with the following differences (lines in blue in the pseudo-code). In Alg. 1, a write( 𝑤 ) obtains a new sequence number 𝑠 + 1 and then attempts to change 𝑅 to ( 𝑠 + 1 , 𝑤, 𝑟𝑎𝑛𝑑 𝑠 + 1 ) . The operation terminates after it succeeds in doing so, or if it sees in 𝑅 a sequence number 𝑠 ′ ≥ 𝑠 + 1. In the latter case, a concurrent write( 𝑤 ′ ) has succeeded and may be seen as overwriting 𝑤 , so write( 𝑤 ) can terminate, even if 𝑤 is never written to 𝑅 . The implementation of writeMax uses a similar idea, except that (1) we make sure that the successive values in 𝑅 are non-decreasing and (2) a writeMax( 𝑤 ) with sequence number 𝑠 + 1 is no longer abandoned when a sequence number 𝑠 ′ ≥ 𝑠 + 1 is read from 𝑅 , but instead when 𝑅 stores a value 𝑤 ′ ≥ 𝑤 .

| Algorithm 2 Auditable Max Register   | Algorithm 2 Auditable Max Register                                                                                                                                                                  |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                      | shared registers 𝑅, SN , 𝑉 [ 0 .. + ∞] , 𝐵 [ 0 .. + ∞][ 0 ..𝑚 - 1 ] as in Alg. 1 𝑀 : a (non-auditable) max register, initially 𝑣 0 = ( 𝑤 0 , local variables: writer, reader, auditor, as in Alg. 1 |
| 21:                                  | 𝑁 0 ) read( ), audit( ): same as in Alg 1                                                                                                                                                           |
|                                      | function                                                                                                                                                                                            |
| 22:                                  | function writeMax( 𝑤 )                                                                                                                                                                              |
| 23:                                  | 𝑣 ←( 𝑤, 𝑁 ) , where 𝑁 is a fresh random nonce                                                                                                                                                       |
| 24:                                  | 𝑀. writeMax ( 𝑣 ) ; 𝑠𝑛 ← SN . read () + 1;                                                                                                                                                          |
| 25:                                  | repeat                                                                                                                                                                                              |
| 26:                                  | ( 𝑙𝑠𝑛, 𝑙𝑣𝑎𝑙, 𝑏𝑖𝑡𝑠 ) ← 𝑅. read ()                                                                                                                                                                    |
| 27:                                  | if 𝑙𝑣𝑎𝑙 ≥ 𝑣 then 𝑠𝑛 ← 𝑙𝑠𝑛 ; break                                                                                                                                                                   |
| 28:                                  | if 𝑙𝑠𝑛 ≥ 𝑠𝑛 then                                                                                                                                                                                    |
| 29:                                  | SN . compare & swap ( 𝑠𝑛 - 1 , 𝑠𝑛 ) ;                                                                                                                                                               |
| 30:                                  | 𝑠𝑛 ← SN . read () + 1; continue                                                                                                                                                                     |
| 31:                                  | 𝑚𝑣𝑎𝑙 ← 𝑀. read ()                                                                                                                                                                                   |
| 32:                                  | 𝑉 [ 𝑙𝑠𝑛 ] . write ( 𝑙𝑣𝑎𝑙.𝑣𝑎𝑙𝑢𝑒 ) ;                                                                                                                                                                  |
| 33:                                  | 𝐵 [ 𝑙𝑠𝑛 ] [ 𝑗 ] . write ( true ) ∀ 𝑗 , s.t. 𝑏𝑖𝑡𝑠 [ 𝑗 ] ≠ 𝑟𝑎𝑛𝑑 𝑙𝑠𝑛 [ 𝑗 ]                                                                                                                             |
| 34:                                  | until 𝑅. compare & swap (( 𝑙𝑠𝑛, 𝑙𝑣𝑎𝑙, 𝑏𝑖𝑡𝑠 ) , ( 𝑠𝑛,𝑚𝑣𝑎𝑙,𝑟𝑎𝑛𝑑 𝑠𝑛                                                                                                                                    |
| 35:                                  | SN . compare & swap ( 𝑠𝑛 - 1 , 𝑠𝑛 ) ; return                                                                                                                                                        |

There is however, a subtlety that must be taken care of. A reader may obtain a value 𝑣 with sequence number 𝑠 , and later read a value 𝑣 + 2 with sequence number 𝑠 ′ &gt; 𝑠 + 1. This leaks to the reader that some writeMax operations occur in between its read operations, and in particular, that a writeMax ( 𝑣 + 1 ) occurred, without ever effectively reading 𝑣 + 1.

To deal with this problem, we append a random nonce 𝑁 to the argument of a writeMax operation, where 𝑁 is a random number. The pair ( 𝑤, 𝑁 ) is used as the value written 𝑣 was used in Alg. 1. The pairs ( 𝑤, 𝑁 ) are ordered lexicographically, that is, first by their value 𝑤 and then by their nonce 𝑁 . Thus, the reader cannot guess intermediate values. The code for read and audit is slightly adjusted in Alg. 2 versus Alg. 1, to ignore the random nonce 𝑁 from the pairs when values are returned.

In the algorithm, a (non-auditable) max-register 𝑀 is shared among the writers. A writeMax( 𝑤 ) by 𝑝 starts by writing the pair 𝑣 = ( 𝑤, 𝑁 ) of the value 𝑤 and the nonce 𝑁 to 𝑀 , before entering a repeat loop. Each iteration is an attempt to store in 𝑅 the current value 𝑚𝑣𝑎𝑙 of 𝑀 , and the loop terminates as soon as 𝑅 holds a value equal to or larger than 𝑚𝑣𝑎𝑙 . Like in Alg. 1, 𝑅 holds a triplet ( 𝑠, 𝑣𝑎𝑙, 𝑏𝑖𝑡𝑠 ) where 𝑠 is 𝑣𝑎𝑙 's sequence number, 𝑣𝑎𝑙 is the current value, and 𝑏𝑖𝑡𝑠 is the encrypted set of readers of 𝑣𝑎𝑙 . Before attempting to change 𝑅 𝑣𝑎𝑙 , and the set of readers, once deciphered, are stored in the registers 𝑉 𝑠 [ ] and 𝐵 𝑠 [ ] , from which they can be retrieved with audit.

In each iteration of the repeat loop, the access pattern of write in Alg. 1 to the shared register SN and 𝑅 is preserved. After obtaining a new sequence number 𝑠 + 1, where 𝑠 is the current value of 𝑆𝑁 (line 24 for the first iteration, line 30 otherwise), a triple ( 𝑙𝑠𝑛, 𝑙𝑣𝑎𝑙, 𝑏𝑖𝑡𝑠 ) is read from 𝑅 . If 𝑙𝑣𝑎𝑙 ≥ 𝑣 , the loop breaks as a value that is equal to or larger than 𝑣 has already been written. As in Alg. 1, before returning we make sure that the sequence number in 𝑆𝑁 is at least as large as 𝑙𝑠𝑛 , the sequence number in 𝑅 .

## 5 AUDITABLE SNAPSHOT OBJECTS AND VERSIONED TYPES

We show how an auditable max register (Section 4) can be used to make other object types auditable.

## 5.1 Making Snapshots Auditable

We start by showing how to implement an auditable 𝑛 -component snapshot object, relying on an auditable max register. Each component has a state, initially ⊥ , and a different designated writer process. A view is an 𝑛 -component array, each cell holding a value written by a process in its component. A atomic object [1] provides two operations: update( 𝑣 ) that changes the process's component to 𝑣 , and scan that returns a view. It is required that in any sequential execution, in the view returned by a scan, each component contains the value of the latest update to this component (or ⊥ if there is no previous update). As for the auditable register, an audit operation returns a set of pairs ( 𝑗, 𝑣𝑖𝑒𝑤 ) . In a sequential execution, there is such a pair if and only if the operation is preceded by a scan by process 𝑝 𝑗 that returns 𝑣𝑖𝑒𝑤 . Here, we want that audits report exactly those scans that have made enough progress to infer the current 𝑣𝑖𝑒𝑤 of the object.

Denysuk and Woeffel [11] show that a strongly-linearizable max register can be used to transform a linearizable snapshot into its strongly linearizable counterpart. As we explain next, with the same technique, non-auditable snapshot objects can be made auditable. Algorithm 3 adds an audit operation to their algorithm. Their implementation is lock-free, as they rely on a lock-free implementation of a max register. Algorithm 3 is wait-free since we use the wait-free max-register implementation of Section 4.

Let 𝑆 be a linearizable, but non-auditable snapshot object. The algorithm works as follows: each new state (that is, whenever one component is updated) is associated with a unique and increasing version number . The version number is obtained by storing a sequence number 𝑠𝑛 𝑖 in each component 𝑖 of 𝑆 , in addition to its current value. Sequence number 𝑠𝑛 𝑖 is incremented each time the 𝑖 th component is updated (line 2). Summing the sequence numbers of the components yields a unique and increasing version number ( 𝑣𝑛 ) for the current view.

The pairs ( 𝑣𝑛, 𝑣𝑖𝑒𝑤 ) , where 𝑣𝑛 is a version number and 𝑣𝑖𝑒𝑤 a state of the auditable snapshot, are written to an auditable max register 𝑀 . The pairs are ordered according to the version number, which is a total order since version numbers are unique. Therefore, the latest state can be retrieved by reading 𝑀 , and the set of past scan operations can be obtained by auditing 𝑀 (line 10). The current view of the auditable snapshot is stored in 𝑆 .

In an update( 𝑣 ), process 𝑝 𝑖 starts by updating the 𝑖 th component of 𝑆 with 𝑣 and incrementing the sequence number field 𝑠𝑛 𝑖 . It then scans 𝑆 , thus obtaining a new view of 𝑆 that includes its update. The view 𝑣𝑖𝑒𝑤 of the implemented auditable snapshot is obtained by removing the sequence number in each component (line 4). The version number 𝑣𝑛 associated with this view is the sum of the sequence numbers. It then writes ( 𝑣𝑛, 𝑣𝑖𝑒𝑤 ) to the max-register 𝑀 (line 5). A scan operation reads a pair ( 𝑣𝑛, 𝑣𝑖𝑒𝑤 ) from 𝑀 and returns the corresponding 𝑣𝑖𝑒𝑤 (line 7). Since 𝑀 is auditable, the views returned by the processes that have previously performed a scan can thus be inferred by auditing 𝑀 (line 10).

The audit and scan operations interact with the implementation by applying a single operation (audit and read, respectively) to the auditable max register 𝑀 . The algorithm therefore lifts the properties of the implementation of 𝑀 to the auditable snapshot object. In particular, when the implementation presented in Section 4 is used, effective scan operations are auditable, scan operations are uncompromised by other scanners, and update operations are uncompromised by scanners.

## 5.2 Proof of Correctness

Let 𝛼 be a finite execution of Algorithm 3. To simplify the proof, we assume the inputs of update by the same process are unique.

| Algorithm 3 𝑛 -component auditable snapshot objects. 1:   | Algorithm 3 𝑛 -component auditable snapshot objects. 1:                                                                                                                                                                                                        | ⊲ code for writer 𝑝 , 𝑖 1 , . . . , 𝑛   |
|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
|                                                           | shared registers M : auditable max register, initially ( 0 , [⊥ , . . . , ⊥]) S : (non-auditable) snapshot object, initially [( 0 , ⊥) , . . . , ( 0 , ⊥)] local variable: writer 𝑝 𝑖 , 1 ≤ 𝑖 ≤ 𝑛 𝑠𝑛 𝑖 local sequence number, initially 0 function update( 𝑣 ) | 𝑖 ∈ { }                                 |
| 2:                                                        | 𝑠𝑛 𝑖 ← 𝑠𝑛 𝑖 + 1; 𝑆. update 𝑖 (( 𝑠𝑛 𝑖 , 𝑣 ))                                                                                                                                                                                                                    |                                         |
| 3:                                                        | 𝑠𝑣𝑖𝑒𝑤 ← 𝑆. scan () ; 𝑣𝑛 ← ˝ 1 ≤ 𝑗 ≤ 𝑛 𝑠𝑣𝑖𝑒𝑤 [ 𝑗 ] .𝑠𝑛 𝑣𝑖𝑒𝑤 ← the 𝑛 -component array of the values in 𝑠𝑣𝑖𝑒𝑤                                                                                                                                                     |                                         |
| 5:                                                        | 𝑀. writeMax (( 𝑣𝑛, 𝑣𝑖𝑒𝑤 )) ; return                                                                                                                                                                                                                            |                                         |
| 6:                                                        | function scan( )                                                                                                                                                                                                                                               |                                         |
| 7:                                                        | ( _ , 𝑣𝑖𝑒𝑤 ) ← 𝑀. read () ; return 𝑣𝑖𝑒𝑤                                                                                                                                                                                                                        |                                         |
| 8:                                                        | function audit( )                                                                                                                                                                                                                                              |                                         |
| 9:                                                        | MA ← 𝑀. audit () ;                                                                                                                                                                                                                                             |                                         |
| 10:                                                       | return {( 𝑗, 𝑣𝑖𝑒𝑤 ) : ∃ an element ( 𝑗, (∗ , 𝑣𝑖𝑒𝑤 )) ∈ MA                                                                                                                                                                                                      |                                         |

Weassumethat the implementation of 𝑀 is wait-free and linearizable. In addition, it guarantees effective linearizability and that read operations are uncompromised by other readers. We also assume that the implementation of 𝑆 is linearizable and wait-free (e.g.,[1]). Inspection of the code shows that update, scan and audit operations are wait-free.

Since

𝑆

and

𝑀

are linearizable and linearizability is composable,

𝛼

can be seen as a sequence of steps applied to

𝑆

or

𝑀

.

In particular, we associate with each high-level operation linearization

𝑜𝑝

a step

𝜎 𝑜𝑝

(

)

applied by

𝑜𝑝

either to

𝑆

or to

𝑀

.

The

𝐿 𝛼

(

)

of

𝛼

is the sequence formed by ordering the operations according to the order their associated step occurs in

𝛼

.

For a scan and an audit operation 𝑜𝑝 , 𝜎 𝑜𝑝 ( ) is, respectively, the read and the audit steps applied to 𝑀 . If 𝑜𝑝 is an update with input 𝑥 by process 𝑝 𝑖 , then let 𝑣𝑛 𝑥 be the sum of the sequence numbers 𝑠𝑛 in each component of 𝑆 after update ( 𝑥 ) has been applied to 𝑆 by 𝑝 𝑖 . 𝜎 𝑜𝑝 ( ) is the first write to 𝑀 of a pair ( 𝑣𝑛, 𝑣𝑖𝑒𝑤 ) with 𝑣𝑛 ≥ 𝑣𝑛 𝑥 and 𝑣𝑖𝑒𝑤 [ 𝑖 ] = 𝑥 . If there is no such write , 𝑜𝑝 is discarded.

We first show that the linearization 𝐿 𝛼 ( ) respects the real-time order between operations.

Lemma 9. If an operation 𝑜𝑝 completes before an operation 𝑜𝑝 ′ is invoked in 𝛼 , then 𝑜𝑝 precedes 𝑜𝑝 ′ in 𝐿 𝛼 ( ) .

Proof. We show that that the linearization point of any operation 𝑜𝑝 is inside its execution interval; the claim is trivial for scan or audit operations.

Suppose that 𝑜𝑝 is an update by a process 𝑝 𝑖 with input 𝑥 . The sum of the sequence numbers in the components of 𝑆 increases each time an update is applied to it. Hence, any pair ( 𝑣𝑛, 𝑣𝑖𝑒𝑤 ) written to 𝑀 before 𝑝 𝑖 has updated its component of 𝑆 to 𝑥 is such that 𝑣𝑛 &lt; 𝑣𝑛 𝑥 . Therefore 𝜎 𝑜𝑝 ( ) , if it exists, is after 𝑜𝑝 starts. If 𝑜𝑝 terminates, then it scans 𝑆 after updating the 𝑖 th component of 𝑆 to 𝑥 . The 𝑣𝑖𝑒𝑤 it obtains and its associated version number satisfy 𝑣𝑖𝑒𝑤 [ 𝑖 ] = 𝑥 and 𝑣𝑛 ≥ 𝑣𝑛 𝑥 . This pair is written to 𝑀 . If 𝜎 𝑜𝑝 ( ) is not this step, then 𝜎 𝑜𝑝 ( ) occurs before 𝑜𝑝 terminates. If 𝑜𝑝 does not terminate and 𝜎 𝑜𝑝 ( ) does exist, it occurs after 𝑜𝑝 starts and thus within 𝑜𝑝 's execution interval. □

Lemma 10. Each component 𝑖 of the view returned by a scan is the input of the last update by 𝑝 𝑖 linearized before the scan in 𝐿 𝛼 ( ) .

Proof. Consider a scan operation 𝑠𝑜𝑝 that returns 𝑣𝑖𝑒𝑤 , with 𝑣𝑖𝑒𝑤 [ 𝑖 ] = 𝑥 . This view is read from the max register 𝑀 and has version number 𝑣𝑛 . Let 𝑜𝑝 be the last update by 𝑝 𝑖 linearized before 𝑠𝑜𝑝 in 𝐿 𝛼 ( ) , let 𝑦 be its input and 𝑣𝑛 𝑦 the version number (that is the sum of the sequence number stored in each component) of 𝑆 immediately after 𝑆. update ( 𝑦 ) is applied by 𝑝 𝑖 .

We denote by 𝜎 𝑢 this low level update . Since the version number increases with each update , every pair ( 𝑣𝑛 , 𝑣𝑖𝑒𝑤 ′ ′ ) written into 𝑀 before 𝜎 𝑢 is such that 𝑣𝑛 ′ &lt; 𝑣𝑛 𝑦 . Also, every pair ( 𝑣𝑛 , 𝑣𝑖𝑒𝑤 ′ ′ ) written to 𝑀 after 𝜎 𝑢 and before 𝑠𝑜𝑝 is linearized satisfies 𝑣𝑛 ′ ≥ 𝑣𝑛 𝑦 = ⇒ 𝑣𝑖𝑒𝑤 ′ [ 𝑖 ] = 𝑦 . Indeed, if 𝑣𝑛 ′ ≥ 𝑣𝑛 𝑦 , 𝑣𝑖𝑒𝑤 ′ is obtained by a scan of 𝑆 applied after the 𝑖 -th component is set to 𝑦 . Hence, 𝑣𝑖𝑒𝑤 ′ [ 𝑖 ] = 𝑦 because we assume that 𝑜𝑝 is the last update by 𝑝 𝑖 linearized before 𝑠𝑜𝑝 in 𝐿 𝛼 ( ) .

Finally, step 𝜎 𝑜𝑝 ( ) is a write of pair ( 𝑣𝑛 , 𝑣𝑖𝑒𝑤 ′ ′ ) to 𝑀 with 𝑣𝑛 ′ ≥ 𝑣𝑛 𝑦 and 𝑣𝑖𝑒𝑤 ′ [ 𝑖 ] = 𝑦 𝜎 𝑜𝑝 . ( ) occurs after 𝜎 𝑢 and before the max register 𝑀 is read by 𝑠𝑜𝑝 . It thus follows that the pair ( 𝑣𝑛, 𝑣𝑖𝑒𝑤 ) read from 𝑀 in 𝑠𝑜𝑝 satisfies 𝑣𝑛 ≥ 𝑣𝑛 𝑦 and has been written after 𝜎 𝑦 . Hence, 𝑣𝑖𝑒𝑤 [ 𝑖 ] = 𝑦 = 𝑥 . We conclude that each component 𝑖 of the view returned by a scan is the input of the last update by 𝑝 𝑖 linearized before the scan in 𝐿 𝛼 ( ) . □

Lemma11. An audit reports ( 𝑗, 𝑣𝑖𝑒𝑤 ) if and only if 𝑝 𝑗 has a 𝑣𝑖𝑒𝑤 -effective 2 scan in 𝛼 . Each update ( 𝑣 ) is uncompromised by a scanner 𝑝 𝑗 unless it has a 𝑣𝑖𝑒𝑤 -effective scan with one component of 𝑣𝑖𝑒𝑤 equal to 𝑣 . Each scan by 𝑝 𝑘 is uncompromised by a scanner 𝑝 𝑗 ≠ 𝑝 𝑘 .

Proof. A scan applies a single operation on shared objects, namely a read on 𝑀 . It is linearized with this step, which determines the view it returns. Therefore, a scan is linearized if and only if it is effective. Hence ( 𝑗, 𝑣𝑖𝑒𝑤 ) is reported by an audit if and only if 𝑝 𝑗 has a 𝑣𝑖𝑒𝑤 -effective scan.

Let 𝑣 be the input of an update operation by some process 𝑝 𝑖 . If there is no 𝑣𝑖𝑒𝑤 with 𝑣𝑖𝑒𝑤 [ 𝑖 ] = 𝑣 written to 𝑀 (line 5), update( 𝑣 ) can be replaced by update( 𝑣 ′ ), 𝑣 ′ ≠ 𝑣 in an execution 𝛼 ′ , 𝛼 𝑝 𝑗 ∼ 𝛼 ′ . Otherwise, note that each 𝑠𝑣𝑖𝑒𝑤 for which 𝑝 𝑗 has a 𝑠𝑣𝑖𝑒𝑤 -effective scan, we have 𝑠𝑣𝑖𝑒𝑤 [ 𝑖 ] ≠ 𝑣 . Suppose that 𝑣𝑖𝑒𝑤 , with 𝑣𝑖𝑒𝑤 [ 𝑖 ] = 𝑣 is written to 𝑀 in 𝛼 . Then we can replace 𝑣𝑖𝑒𝑤 with an array 𝑣𝑖𝑒𝑤 ′ , identical to 𝑣𝑖𝑒𝑤 except that 𝑣𝑖𝑒𝑤 ′ [ 𝑖 ] = 𝑣 ′ ≠ 𝑣 an execution 𝛼 ′ 𝑝 𝑗 ∼ 𝛼 . This is because the write of 𝑣𝑖𝑒𝑤 is not compromised by 𝑝 𝑗 in 𝑀 . By repeating this procedure until all writes to 𝑀 of 𝑣𝑖𝑒𝑤 s with 𝑣𝑖𝑒𝑤 [ 𝑖 ] = 𝑣 have been eliminated leads to an execution 𝛽, 𝛽 𝑝 𝑗 ∼ 𝛼 in which there is no update( 𝑣 ). □

Theorem 12. Alg. 3 is a wait-free linearizable implementation of an auditable snapshot object which audits effective scan operations, in which scan and update are uncompromised by scanners.

## 5.3 Versioned Objects

Snapshot objects are an example of a versioned type [11], whose successive states are associated with unique and increasing version numbers. Furthermore, the version number can be obtained from the object itself, without resorting to external synchronization primitives. Essentially the same construction can be applied to any versioned object.

An object 𝑡 ∈ T is specified by a tuple ( 𝑄,𝑞 , 𝐼, 𝑂, 𝑓 , 𝑔 0 ) , where 𝑄 is the state space, 𝐼 and 𝑂 are respectively the input and output sets of update and read operations. 𝑞 0 is the initial state and functions 𝑓 : 𝑄 → 𝑂 and 𝑔 : 𝐼 × 𝑄 → 𝑄 describes the sequential behavior of read and update . A read () operation leaves the current state 𝑞 unmodified and returns 𝑓 ( 𝑞 ) . An update ( 𝑣 ) , where 𝑣 ∈ 𝐼 changes the state 𝑞 to 𝑔 𝑣, 𝑞 ( ) and does not return anything.

A linearizable versioned implementation of a type 𝑡 ∈ T can be transformed into a strongly-linearizable one [11], as follows. Let 𝑡 = ( 𝑄,𝑞 , 𝐼, 𝑂, 𝑓 , 𝑔 0 ) be some type in T . Its versioned variant 𝑡 ′ = ( 𝑄 ,𝑞 , 𝐼 ′ ′ 0 ′ , 𝑂 ′ , 𝑓 ′ , 𝑔 ′ ) has 𝑄 ′ = 𝑄 × N ,

2 Namely, 𝑝 𝑖 has a scan operation that returns 𝑣𝑖𝑒𝑤 in all indistinguishable executions.

𝑞 ′ 0 = ( 𝑞 , 0 0 , ) 𝐼 ′ = 𝐼 , 𝑂 ′ = 𝑂 × N , 𝑓 ′ : 𝑄 ′ → 𝑂 × N and 𝑔 ′ : 𝐼 × 𝑄 ′ → 𝑄 ′ . That is, the state of 𝑡 ′ is augmented with a version number, which increases with each update and is returned by each read : 𝑓 ′ (( 𝑞, 𝑣𝑛 )) = ( 𝑓 ( 𝑞 , 𝑣𝑛 ) ) and 𝑔 ′ (( 𝑞, 𝑣𝑛 )) = ( 𝑔 𝑞 , 𝑣𝑛 ( ) ′ ) with 𝑣𝑛 &lt; 𝑣𝑛 ′ .

A versioned implementation of a type 𝑡 ∈ T can be transformed into an auditable implementation of the same type using an auditable register. The construction is essentially the same as presented in Algorithm 3. In the auditable variant 𝑇 𝑎 of 𝑇 , to perform an update( 𝑣 ), a process 𝑝 first update the versioned implementation 𝑇 before reading it. 𝑝 hence obtains a pair ( 𝑜, 𝑣𝑛 ) that it writes to the auditable max register 𝑀 . For a read, a process returns what it reads from 𝑀 . As read amounts to read 𝑀 , to perform an audit a process simply audit the max-register 𝑀 . As we have seen for snapshots, 𝑇 𝑎 is linearizable and wait-free. Moreover, 𝑇 𝑎 inherits the advanced properties of the underlying max-register: If 𝑀 is implemented with Algorithm 2, then it correctly audits effective read, and read and update are uncompromised.

Theorem 13 (versioned types are auditable). Let 𝑡 ∈ T , and let 𝑇 be a versioned implementation of 𝑡 that is linearizable and wait-free. There exists a wait-free, linearizable and auditable implementation of 𝑡 from 𝑇 and auditable max-registers in which read and update are uncompromised by readers and audit reports only effective read operations.

## 6 DISCUSSION

This paper introduces novel notions of auditability that deal with curious readers. We implement a wait-free linearizable auditable register that tracks effective reads while preventing unauthorized audits by readers. This implementation is extended into an auditable max register, which is then used to implement auditable atomic snapshots and versioned types.

Many open questions remain for future research. An immediate question is how to implement an auditable register in which only auditors can audit , i.e., reads are uncompromised by writers. A second open question is how to extend auditing to additional objects. These can include, for example, partial snapshots [4] in which a reader can obtain an 'instantaneous' view of a subset of the components. Another interesting object is a clickable atomic snapshot [16], in particular, variants that allow arbitrary operations on the components and not just simple updates (writes).

The property of uncompromising other accesses can be seen as an internal analog of history independence , recently investigated for concurrent objects [3]. A history-independent object does not allow an external observer, having access to the complete system state , to learn anything about operations applied to the object, but only its current state. Our definition, on the other hand, does not allow an internal observer, e.g., a reader that only reads shared base objects, to learn about other read and write operations applied in the past. An interesting intermediate concept would allow several readers collude and to combine the information they obtain in order to learn more than what they are allowed to.

## ACKNOWLEDGMENTS

H. Attiya is supported by the Israel Science Foundation (grant number 22/1425). A. Fernández Anta has been funded by project PID2022-140560OB-I00 (DRONAC) funded by MICIU / AEI / 10.13039 / 501100011033 and ERDF, EU. A. Milani is supported by the France 2030 ANR project ANR-23-PECL-0009 TRUSTINCloudS. C. Travers is supported in part by ANR projects DUCAT (ANR-20-CE48-0006).
<|endofpaper|>