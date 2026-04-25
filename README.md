
Neuromorphic Computing 

Vishal Wagh 

Purdue University 

December 6, 2025 

Abstract           

Modern deep learning has produced transformative capabilities, but current large models 
and conventional computing architectures face important limitations: high energy use, high 
latency in on-device contexts, and poor alignment with how biological brains compute. 
Neuromorphic computing and spiking neural networks (SNNs) are brain-inspired approaches 
that promise event-driven, low-power, and temporally aware computation. This literature review 
synthesizes technical research across three themes: (1) shortcomings of the von Neumann model 
when applied to AI workloads, (2) adaptability and training challenges specific to SNNs 
(including surrogate gradient methods, STDP, and ANN→SNN conversion), and (3) practical 
obstacles for neuromorphic hardware adoption (programming ecosystems, mapping algorithms 
to silicon, device non-idealities, and economics). The review integrates results from recent 
surveys and research papers to highlight where progress has been made and where gaps remain. 
The synthesis indicates that algorithm-hardware co-design, improved training methods, robust 
benchmarking, and open toolchains are necessary to realize neuromorphic systems at scale.  
Keywords: neuromorphic computing, spiking neural networks, von Neumann, surrogate 
gradients, hardware–software co-design. 


Introduction 

Artificial intelligence has made rapid progress over the last decade, driven primarily by 
large artificial neural network (ANN) models running on traditional von Neumann hardware 
(CPUs/GPUs). However according to Kimovski et al. (2024), while this approach has produced 
strong performance on many benchmarks, the prevailing compute model exhibits several 
structural weaknesses for the next generation of AI applications: data movement between 
memory and processor causes a power and performance bottleneck; synchronous, dense compute 
is inefficient for sparse, event-driven workloads; and present hardware lacks native support for 
temporal/spike-based computation inspired by biology. These constraints motivate the 
investigation of neuromorphic computing: hardware and algorithms that mimic neuronal and 
synaptic dynamics, executing spike-based networks with high parallelism and low power 
consumption (Al Abdul Wahid et al., 2024). 
This literature review focuses on the technical intersection of Spiking Neural Networks 
(SNNs) and Neuromorphic Computing (NC). It is organized around three themes. Theme 1 
reviews the principal limitations of traditional von Neumann architectures for AI workloads. 
Theme 2 surveys the adaptability and training challenges of SNNs, including methods developed 
to close the gap between SNNs and ANNs. Theme 3 addresses hardware and system challenges 
associated with neuromorphic computing platforms, including mapping algorithms to silicon and 
production-level constraints. Taken together, the evidence suggests that algorithm–hardware co
design, open and reproducible toolchains, and energy-aware benchmarks are the most actionable 
priorities. Advancing these areas will make neuromorphic systems more competitive for 
production deployments in autonomous systems, robotics, and always-on edge sensing. 


Limitations of Traditional Computing 

Modern computing systems are dominated by a Von Neumann architecture that separates 
memory and computation. Muralidhar et al. (2022) describes the von Neumann bottleneck as 
the cost (time and energy) of shuttling data between memory and processor, this becomes a 
primary limiter when workloads are memory-bound or when models require vast parameter 
accesses. In case of AI, especially large models and streaming sensory processing, the cost of 
memory movement can dwarf arithmetic costs, increasing latency and energy expenditure. 
Another limitation is the mismatch between synchronous dense computation on GPUs 
and the sparse, asynchronous nature of many real-world signals. For applications that produce 
sparse events (e.g. changes in a scene captured by event-based cameras), traditional frame-based 
processing wastes computation on unchanged regions and cannot take full advantage of temporal 
sparsity. In contrast, spike-based processing is inherently event-driven and can avoid 
unnecessary operations. 
Thermal and energy constraints also shape system design. Kimovski et al. (2024) notes 
high-performance GPUs expend significant energy and produce heat, challenging continuous on
device operation for robotics, battery-powered sensors, and edge devices. Muralidhar et al. 
(2022) add that such physical constraints motivate computing models that merge memory and 
computation and leverage sparse event processing to reduce energy per useful operation. 
Thus, according to Davies et al. (2021) biological plausibility is a practical consideration, 
because biological nervous systems perform robust, online, low-power processing using local 
plasticity rules and massively parallel sparse communication. Translating these principles into 
silicon offers potential efficiency and latency benefits, but doing so requires departing from von 
Neumann design assumptions. The literature consistently argues that certain future AI domains 
such as real-time robotics, always-on sensing, and adaptive control will be better supported by 
neuromorphic paradigms than by traditional architectures (Chowdhury et al., 2025). 
Synthesis of Limitations of Traditional Computing highlights that -current AI is limited 
by the foundational structure of von Neumann computing, overcoming them calls for 
architectures that minimize memory-movement, embrace asynchronous event processing, and 
co-locate memory and computation which precisely the goals of neuromorphic approaches, 
which we’ll discuss in the next theme. 
Adaptability and Training Challenges of Spiking Neural Networks (SNNs) 
SNNs model information using discrete spikes and time; neurons fire only when 
membrane potentials cross thresholds, and timing encodes information. This temporal, event
driven representation is a key advantage for low-latency and energy-efficient processing. 
Nevertheless, SNNs face significant training and adaptability challenges that have prevented 
them from easily attaining the performance of modern ANNs on large benchmarks. 

I. 
Training difficulty and non-differentiability 
Standard gradient-based learning relies on differentiable activation functions. Spikes are 
discontinuous events, and thus direct backpropagation through spiking dynamics is not 
straightforward. However there are several approaches have been proposed. 
Surrogate gradient methods by Gouda et al. (2024) and Li et al. (2024), where we replace 
the non-differentiable spike function with a smooth surrogate derivative during the backward 
pass, allowing gradient descent training of SNNs end-to-end. Surrogate gradients have enabled 
deeper spiking architectures to be trained and have improved task performance but tuning and 
stability remain areas of active research. On contrast, Gao et al. (2023) and Wang et al. (2024) 
proposed ANN-to-SNN conversion -train an ANN with mature techniques, then convert 
activations/weights to spike rates in an SNN. This approach often preserves accuracy but can 
require long simulation windows to approximate real-valued activations, which reduces latency 
advantages.  

II. 
Scalability and performance gaps 
Despite progress, SNNs often trail ANNs on large, complex datasets when trained from 
scratch. This gap is narrowing with surrogate gradients and hybrid schemes, but Zhou et al. 
(2024) from Frontiers review notes that achieving parity without sacrificing latency and energy 
benefits is still a research target. The literature suggests that new spiking architectures, not mere 
one-to-one translations of ANN topologies, are needed to exploit temporal coding fully. 

III. 
Tooling and frameworks 
Zhou et al. (2024) highlights one of the main obstacles in SNN adoption -the fragmented 
SNN ecosystem. SNNs do have simulation frameworks such as Brian2, Nengo, BindsNET, and 
SNN libraries but they are built atop PyTorch, but there is no single dominant, user-friendly 
ecosystem comparable to TensorFlow or PyTorch built for ANNs. This fragmentation 
complicates reproducibility and slows adoption by mainstream ML practitioners. 
This synthesis highlights that training and adaptability challenges for SNNs are both 
algorithmic (non-differentiability, architecture design) and infrastructural (tooling, benchmarks). 
Progress in surrogate gradients, hybrid training, and community toolchains is promising, but 
converging to robust, scalable methods that preserve SNNs’ energy/latency advantages remains a 
primary research priority. The next theme discusses how Neuromorphic hardware compliments 
SNNs in solving the limitations of Von Neumann computing. 



Challenges with Neuromorphic Computing 
Neuromorphic computing seeks to realize SNN advantages in hardware. Several 
commercially developed and research platforms exist (Intel Loihi family, IBM TrueNorth, 
SpiNNaker, Tianjic, BrainScaleS), each reflecting different trade-offs between programmability, 
energy, and scalability (Davies et al., 2021; Zhang et al., 2023). Nonetheless, bringing 
neuromorphic hardware into widespread use faces multiple obstacles. 

I. 
Hardware heterogeneity and device non-idealities 
Neuromorphic platforms vary: some are digital (Loihi), some analog or mixed-signal, and 
emerging devices (memristors, spintronics, photonic devices) promise higher density and lower 
power but introduce device variability and analog mismatch issues (Muir et al., 2025). Device 
non-idealities (noise, drift, limited precision) complicate mapping algorithms and require 
resilience at the algorithm level. 

II. 
Software and programming model fragmentation 
Davies et al, (2021) add that there is no unified programming model for neuromorphic systems. 
Porting an SNN from a simulator to a specific chip often requires substantial modification and 
hardware-aware mapping. This raises a steep learning curve for developers and reduces 
reproducibility. Middleware and abstraction layers that present consistent APIs to researchers are 
beginning to emerge but are not yet mature. 

III. 
Economic and manufacturing issues 
Most neuromorphic devices are research prototypes or low-volume products. Scaling 
manufacturing to commercial volumes, reducing cost, and building a supporting ecosystem 
(tools, libraries, trained engineers) are nontrivial challenges. The commercialization path requires 
demonstrating clear application value (e.g., energy savings at scale, unique capabilities) that 
justify investment (Muir et al., 2025). 


IV. 
Benchmarking and evaluation 
Al Abdul Wahid et al. (2024) add, the field lacks standardized, energy-aware benchmarks for 
SNN/neuromorphic systems. Benchmarks that combine accuracy, latency, and energy per 
inference - and that are compatible with event-based tasks - are essential to compare approaches 
fairly and to drive optimization. 
Synthesis of Neuromorphic hardware offers strong theoretical benefits, but to move from lab 
prototypes to production systems the community must address heterogeneity and device 
reliability, create unified programming abstractions, optimize mapping and partitioning, and 
establish standardized benchmarks and metrics that reflect operational constraints (energy, 
latency, robustness). 




Conclusion 
Neuromorphic computing and spiking neural networks present a promising route toward 
energy-efficient, temporally sensitive AI suited for real-time and edge applications. The literature 
shows clear conceptual advantages: event-driven processing, the ability to exploit temporal 
coding, and potential orders-of-magnitude reductions in energy per operation. However, the path 
to broad practical adoption traverses significant obstacles in three areas: first, the structural 
limitations of von Neumann hardware for AI workloads motivate alternative architectures; 
second, SNNs still contend with training and scalability problems that require improved 
surrogate gradient techniques, hybrid training, and novel architectures; third, neuromorphic 
hardware must become more accessible and reliable through standardized software stacks, robust 
mapping methods, device engineering, and benchmarking. 
Taken together, the evidence suggests that algorithm–hardware co-design, open and reproducible 
toolchains, and energy-aware benchmarks are the most actionable priorities. Advancing these 
areas will make neuromorphic systems more competitive for production deployments in 
autonomous systems, robotics, and always-on edge sensing. The interdisciplinary nature of this 
research - spanning device physics, VLSI, machine learning, and computational neuroscience - 
requires collaborative teams and coordinated community standards to realize neuromorphic 
computing’s promise. 



              
References 

Al Abdul Wahid, S., Alazzawi, A., & Makeen, S. (2024). A survey on neuromorphic architectures 
for running spiking neural networks. Electronics, 13(15), 2963. 
https://doi.org/10.3390/electronics13152963 
Chowdhury, S. S., et al. (2025). Neuromorphic computing for robotic vision: algorithms to 
systems. Nature Machine Intelligence, 7, 1–13. https://doi.org/10.1038/s44172-025
00492-5 
Davies, M., Wild, A., Orchard, G., Sandamirskaya, Y., Guerra, G., Joshi, P., Plank, P., & Risbud, 
S. R. (2021). Advancing neuromorphic computing with Loihi: A survey of results and 
outlook. Proceedings of the IEEE, 109(5), 911–934. 
https://doi.org/10.1109/JPROC.2021.3067593 
Gouda, M., et al. (2024). Surrogate gradient learning in spiking networks trained on temporal 
tasks. Optics Express, 32(9), 16260–16286. https://doi.org/10.1364/OE.32.016260 
Gao, H., et al. (2023). High-accuracy deep ANN-to-SNN conversion using efficient rate and 
temporal encodings. Frontiers in Neuroscience. 
https://doi.org/10.3389/fnins.2023.1141701 
Kimovski, D., et al. (2024). Beyond von Neumann in the computing continuum. Proceedings / 
Internet Computing / AtLarge Research. (2024). [PDF] https://atlarge
research.com/pdfs/2024-internetcomp-beyond-von-neumann.pdf 
Neuromorphic Computing                
11 
Li, Y., et al. (2024). Directly training temporal spiking neural networks with masked surrogate 
gradients. IEEE Transactions on Neural Networks and Learning Systems. (2024). 
https://doi.org/10.1016/j.neunet.2024.106499 
Muralidhar, R., et al. (2022). Energy efficient computing systems: architectures and methods. 
ACM Computing Surveys / Communications of the ACM. (2022). 
https://doi.org/10.1145/3511094 
Muir, D. R., et al. (2025). The road to commercial success for neuromorphic computing. Nature 
Communications. (2025). https://doi.org/10.1038/s41467-025-57352-1 
Wang, Y., et al. (2024). A universal ANN-to-SNN framework for achieving high performance. 
IEEE Transactions on Neural Networks and Learning Systems. (2024). 
https://doi.org/10.1016/j.neunet.2024.106244 
Zhang, L., et al. (2023). Hardware implementation of spike-based neuromorphic systems: a 
review. Frontiers in Neuroscience / PMC. https://doi.org/10.3389/fnins.2022.1113983 
Zhou, C., et al. (2024). Direct training of high-performance deep spiking neural networks: 
theory, models and future directions. Frontiers in Neuroscience, 18, 1383844. 
https://doi.org/10.3389/fnins.2024.1383844 
