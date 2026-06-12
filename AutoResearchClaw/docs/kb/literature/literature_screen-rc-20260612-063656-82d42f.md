---
created: '2026-06-12T06:44:39+00:00'
evidence:
- stage-05/shortlist.jsonl
id: literature_screen-rc-20260612-063656-82d42f
run_id: rc-20260612-063656-82d42f
stage: 05-literature_screen
tags:
- literature_screen
- stage-05
- run-rc-20260
title: 'Stage 05: Literature Screen'
---

# Stage 05: Literature Screen

{"paper_id": "oalex-W3204628099", "title": "100,000-spin coherent Ising machine", "year": 2021, "venue": "Science Advances", "citation_count": 323, "doi": "10.1126/sciadv.abh0952", "arxiv_id": "", "url": "https://doi.org/10.1126/sciadv.abh0952", "source": "openalex", "cite_key": "honjo2021spin", "relevance_score": 0.88, "quality_score": 0.96, "keep_reason": "Core CIM hardware paper for MAX-CUT. Explicitly reports that operating the CIM 'near the phase transition point' (i.e., pump-rate threshold) changes solution quality/distribution — directly informs pump-power scheduling. Top venue, high impact."}
{"paper_id": "oalex-W3035347013", "title": "Coherent Ising machines—Quantum optics and neural network Perspectives", "year": 2020, "venue": "Applied Physics Letters", "citation_count": 65, "doi": "10.1063/5.0016140", "arxiv_id": "", "url": "https://doi.org/10.1063/5.0016140", "source": "openalex", "cite_key": "yamamoto2020coherent", "relevance_score": 0.91, "quality_score": 0.86, "keep_reason": "Directly discusses CIM behavior as the pump rate is raised from below to above threshold and the resulting trapping in local minima — exactly the pump-power dynamics targeted by the topic. Foundational CIM author."}
{"paper_id": "oalex-W3197467828", "title": "Coherent Ising Machines with Optical Error Correction Circuits", "year": 2021, "venue": "Advanced Quantum Technologies", "citation_count": 47, "doi": "10.1002/qute.202100077", "arxiv_id": "", "url": "https://doi.org/10.1002/qute.202100077", "source": "openalex", "cite_key": "reifenstein2021coherent", "relevance_score": 0.83, "quality_score": 0.8, "keep_reason": "Describes CIM with programmable pump amplitude and error-correction feedback derived from the truncated Wigner SDE — relevant to designing/optimizing the pump (gain) schedule for escaping local minima."}
{"paper_id": "oalex-W4313417335", "title": "Speed-up coherent Ising machine with a spiking neural network", "year": 2022, "venue": "Optics Express", "citation_count": 48, "doi": "10.1364/oe.479903", "arxiv_id": "", "url": "https://doi.org/10.1364/oe.479903", "source": "openalex", "cite_key": "lu2022speedup", "relevance_score": 0.78, "quality_score": 0.8, "keep_reason": "Addresses the central CIM problem of local-minima trapping (the same issue pump-power scheduling targets) via added dissipative pulses to the OPO/measurement-feedback CIM. Method directly applicable to CIM dynamics tuning."}
{"paper_id": "oalex-W4322621771", "title": "Recent progress on coherent computation based on quantum squeezing", "year": 2023, "venue": "AAPPS bulletin", "citation_count": 54, "doi": "10.1007/s43673-023-00077-4", "arxiv_id": "", "url": "https://doi.org/10.1007/s43673-023-00077-4", "source": "openalex", "cite_key": "lu2023recent", "relevance_score": 0.76, "quality_score": 0.78, "keep_reason": "Recent review of OPO-based CIM hardware (delayed-path and measurement-feedback schemes), covering operating principles near/above threshold — useful background and design context for pump-power optimization."}
{"paper_id": "oalex-W3214010870", "title": "All-Optical Scalable Spatial Coherent Ising Machine", "year": 2021, "venue": "Physical Review Applied", "citation_count": 40, "doi": "10.1103/physrevapplied.16.054022", "arxiv_id": "", "url": "https://doi.org/10.1103/physrevapplied.16.054022", "source": "openalex", "cite_key": "strinati2021alloptical", "relevance_score": 0.74, "quality_score": 0.82, "keep_reason": "CIM realization whose collective nonlinear oscillator dynamics drive the system toward the Ising ground state — pump/gain ramp is integral to this dynamics, making it relevant to the optimization topic."}
{"paper_id": "oalex-W4386167200", "title": "A spinwave Ising machine", "year": 2023, "venue": "Communications Physics", "citation_count": 60, "doi": "10.1038/s42005-023-01348-0", "arxiv_id": "", "url": "https://doi.org/10.1038/s42005-023-01348-0", "source": "openalex", "cite_key": "litvinenko2023spinwave", "relevance_score": 0.58, "quality_score": 0.8, "keep_reason": "Time-multiplexed Ising-machine variant explicitly motivated by reducing CIM power consumption while solving MAX-CUT. Moderately relevant: power/energy is the optimization axis, though it is not an OPO-pump scheme specifically."}
{"paper_id": "oalex-W3082340655", "title": "Optimization by Neural Networks in the Coherent Ising Machine and its Application to Wireless Communication Systems", "year": 2020, "venue": "IEICE Transactions on Communications", "citation_count": 30, "doi": "10.1587/transcom.2020nvi0002", "arxiv_id": "", "url": "https://doi.org/10.1587/transcom.2020nvi0002", "source": "openalex", "cite_key": "hasegawa2020optimization", "relevance_score": 0.62, "quality_score": 0.68, "keep_reason": "Introduces the CIM and its mutual-coupling optimization scheme for minimizing the Ising Hamiltonian. The wireless part is only an application example; the CIM optimization methodology is on-topic, so kept at moderate relevance."}
{"paper_id": "oalex-W431959182

... (truncated, see full artifact)
