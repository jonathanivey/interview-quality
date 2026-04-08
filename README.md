
# What Makes a Good Response? An Empirical Analysis of Quality in Qualitative Interviews

This repository contains the code for the paper:
> Jonathan Ivey, Anjalie Field, and Ziang Xiao. 2026. What Makes a Good Response? An Empirical Analysis of Quality in Qualitative Interviews. In *arXiv preprint* ArXiv:XXXX.XXXXX [cs].

In this paper we identify, implement, and evaluate 10 proposed measures of interview response quality to determine which are actually predictive of a response's contribution to the study findings. To conduct our analysis, we introduce the [Qualitative Interview Corpus](https://data.qdr.syr.edu/dataset.xhtml?persistentId=doi:10.5064/F6JWVCH6), a newly constructed dataset of 343 interview transcripts with 16,940 participant responses from 14 real research projects.

## Citing Our Work
If you use our work, please cite it as:

```
@misc{ivey2026makesgoodresponseempirical,
      title={What Makes a Good Response? An Empirical Analysis of Quality in Qualitative Interviews}, 
      author={Jonathan Ivey and Anjalie Field and Ziang Xiao},
      year={2026},
      eprint={2604.05163},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.05163}, 
}
```

If you use the Qualitative Interview Corpus, please ensure you cite both our work and the original data deposits. To do this, you can use the following example citation and provided BibTeX.

Example of an ACL style citation:

```
We use the Qualitative Interview Corpus \citep{ivey2026makesgoodresponseempirical}, which contains interview transcripts from 14 research projects \citep{F68TOJJY_2024, F6AGWUJG_2021, F6FYZITI_2025, F6HYTYIJ_2025, F6L9HHYL_2025, F6MTPVK7_2025, F6AHDRFQ_2020, F6QHVGUI_2023, F6R7J9HL_2025, F6UXQABW_2024, F6XZV8BZ_2025, F6YMWPUX_2021, F6Z82KER_2024, F6ZP448B_2017}.

```

BibTeX with our work and the original data deposits:

```
@misc{ivey2026makesgoodresponseempirical,
      title={What Makes a Good Response? An Empirical Analysis of Quality in Qualitative Interviews}, 
      author={Jonathan Ivey and Anjalie Field and Ziang Xiao},
      year={2026},
      eprint={2604.05163},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.05163}, 
}

@data{F68TOJJY_2024,
author = {Steinberg, Beth and Mulugeta, Yulia and Quatman-Yates, Catherine and Williams, Maeghan and Gogineni, Anvitha and Klatt, Maryanna},
publisher = {QDR Main Collection},
title = {{Data for: Barriers and Facilitators to Implementation of Mindfulness in Motion for Firefighters and Emergency Medical Service Providers}},
year = {2024},
version = {V1},
doi = {10.5064/F68TOJJY},
url = {https://doi.org/10.5064/F68TOJJY}
}

@data{F6AGWUJG_2021,
author = {Shuman, Andrew},
publisher = {QDR Main Collection},
title = {{Data for: Drug Shortage Management: A Qualitative Assessment of a Collaborative Approach}},
year = {2021},
version = {V2},
doi = {10.5064/F6AGWUJG},
url = {https://doi.org/10.5064/F6AGWUJG}
}

@data{F6FYZITI_2025,
author = {Alvarez, Carmen},
publisher = {QDR Main Collection},
title = {{Data for: Experiences of Ghanaian Frontline Healthcare Workers During the COVID-19 Pandemic and Healthcare Leadership Recommendations}},
year = {2025},
version = {V1},
doi = {10.5064/F6FYZITI},
url = {https://doi.org/10.5064/F6FYZITI}
}

@data{F6HYTYIJ_2025,
author = {Micatka, Nathan K.},
publisher = {QDR Main Collection},
title = {{Data for: Socializing Policy Feedback: The Persistent Effects of Adolescent Policy Program Use on Political Behaviors and Attitudes in Adulthood}},
year = {2025},
version = {V1},
doi = {10.5064/F6HYTYIJ},
url = {https://doi.org/10.5064/F6HYTYIJ}
}

@data{F6L9HHYL_2025,
author = {Ruedin, Didier and Murahwa, Brian},
publisher = {QDR Main Collection},
title = {{Perspectives on Political Representation}},
year = {2025},
version = {V1},
doi = {10.5064/F6L9HHYL},
url = {https://doi.org/10.5064/F6L9HHYL}
}

@data{F6MTPVK7_2025,
author = {Mersha, Girmay Ayana},
publisher = {QDR Main Collection},
title = {{Data for: Lessons Learned from Operationalizing the Integration of Nutrition-Specific and Nutrition-Sensitive Interventions in Rural Ethiopia}},
year = {2025},
version = {V1},
doi = {10.5064/F6MTPVK7},
url = {https://doi.org/10.5064/F6MTPVK7}
}

@data{F6AHDRFQ_2020,
author = {Fosher, Kerry},
publisher = {QDR Main Collection},
title = {{Marine Corps Staff Noncommissioned Officer Enlisted Education Project}},
year = {2020},
version = {V2},
doi = {10.5064/F6AHDRFQ},
url = {https://doi.org/10.5064/F6AHDRFQ}
}

@data{F6QHVGUI_2023,
author = {Milman, Anita},
publisher = {QDR Main Collection},
title = {{Ascertaining Intergovernmental Coordination Mechanisms}},
year = {2023},
version = {V1},
doi = {10.5064/F6QHVGUI},
url = {https://doi.org/10.5064/F6QHVGUI}
}

@data{F6R7J9HL_2025,
author = {Bezabih, Alemitu Mequanint and Smith, C. Estelle},
publisher = {QDR Main Collection},
title = {{Expanding Models of Delivery for Online Spiritual Care}},
year = {2025},
version = {V1},
doi = {10.5064/F6R7J9HL},
url = {https://doi.org/10.5064/F6R7J9HL}
}

@data{F6UXQABW_2024,
author = {Gazaway, Shena and Wells, Rachel and Haley, John and Gutierrez, Orlando M. and Nix-Parker, Tamara and Martinez, Isaac and Lyas, Clare and Lang-Lindsey, Katina and Knight, Richard and Odom, J. Nicholas},
publisher = {Palliative Care Research Cooperative QDR},
title = {{Exploring the Acceptability of a Community-Enhanced Intervention to Improve Decision Support Partnership between Patients with Chronic Kidney Disease and Their Family Caregivers}},
year = {2024},
version = {V1},
doi = {10.5064/F6UXQABW},
url = {https://doi.org/10.5064/F6UXQABW}
}

@data{F6XZV8BZ_2025,
author = {Furlong, Darcy E. and Romero, Anna and Helström, Kirstin and Lester, Jessica Nina and Karcher, Sebastian},
publisher = {QDR Main Collection},
title = {{Data for: Teaching with Shared Data for Learning Qualitative Data Analysis: A Multi-Sited Case Study of Instructor and Student Experiences}},
year = {2025},
version = {V1},
doi = {10.5064/F6XZV8BZ},
url = {https://doi.org/10.5064/F6XZV8BZ}
}

@data{F6YMWPUX_2021,
author = {Harrison, Krista},
publisher = {Palliative Care Research Cooperative QDR},
title = {{Advance Care Planning in Hospice Organizations: A Qualitative Pilot Study}},
year = {2021},
version = {V2},
doi = {10.5064/F6YMWPUX},
url = {https://doi.org/10.5064/F6YMWPUX}
}

@data{F6Z82KER_2024,
author = {Vignola, Emilia F. and Ahonen, Emily Q. and Hajat, Anjum},
publisher = {QDR Main Collection},
title = {{Data for: What Extraordinary Times Tell Us about Ordinary Ones: A Multiple Case Study of Precariously Employed Food Retail and Service Workers in Two U.S. State Contexts during the COVID-19 Pandemic}},
year = {2024},
version = {V1},
doi = {10.5064/F6Z82KER},
url = {https://doi.org/10.5064/F6Z82KER}
}

@data{F6ZP448B_2017,
author = {O'Neill, Maureen},
publisher = {QDR Main Collection},
title = {{High performance school-age athletes at Australian schools: A study of conflicting demands}},
year = {2017},
version = {V2},
doi = {10.5064/F6ZP448B},
url = {https://doi.org/10.5064/F6ZP448B}
}

```


## Reproducability Instructions

This file contains instructions for how to reproduce the results from our paper. The simplest way to do this is to download the [Qualitative Interview Corpus](https://data.qdr.syr.edu/dataset.xhtml?persistentId=doi:10.5064/F6JWVCH6) and place it in the data folder. With that data you can:

1. Reproduce the results and figures from the paper (start at [Reproducing Results](#reproducing-results)).

2. Reproduce the measurements of response characteristics (start at [Reproducing Measures](#reproducing-measures)).

Or if you want to reproduce everything from the raw data, start at [Recreating The Qualitative Interview Corpus](#recreating-the-qualitative-interview-corpus).

### Reproducing Results

1. **Annotator Agreement:** Use  `annotator-agreement.ipynb` to compare agreement.

2. **Mixed-Effects Regression:**
Use `correlation_matrix.ipynb` to reproduce the correlation matrix image, and use `response_regression.R` to rerun the regression.

3. **Case Study:**
Use `case_study.ipynb` to recreate analysis.

### Reproducing Measures
1. Run all scripts in `reference-free_metrics` to measure the response characteristics.
2. Run `reference-based_metrics/judge_inclusion.py` to measure the quality criterion.
3. Run `interviewer_metrics/judge_techniques.py` to identify interviewer techniques used.
4. Run `data_processing/aggregate_inclusion_and_rq_relevance.py` to get our final scores for the quality criterion and research question relevance.
5. Run `data_processing/merge_response_regression_inputs.py` to merge all of the measures into a single usable file.
6. Run `data_processing/produce_final_data.py` to create the final combined datafile.
7. Run `data_processing/match_annotation_examples.py` to identify the appropriate examples for the annotator agreement comparison.

### Recreating The Qualitative Interview Corpus
1. Download each data deposit from the [Qualitative Data Repository](https://qdr.syr.edu/) using the references in the paper. 

2. Unzip each deposit file and place them in `data/raw_data`.

3. Run every script in `transcript_preprocessing` to process the raw pdf files.

4. Run `data_processing/construct_excerpts.py` to construct the excerpts



