# **BHASHA: Achieving Sarcasm Interpreted Translation**

## **1. Introduction**

Considering the current popularity of social media platforms, informal communication has surged. Often, these platforms host multilingual conversations, and to top it off, much of the language is **informal** and **riddled with sarcasm**.

However, current translation tools struggle to accurately convey the intended meaning of sarcastic messages (see *Figure 1*). This is due to their inability to grasp the nuances of informal language and the **layered meanings** behind sarcasm.

> This research aims to **bridge this gap** in the translation experience by developing a novel approach to translating sarcastic English tweets into **honest Telugu interpretations**.

We designed a **two-pipeline approach** (see *Figure 3*). Our focus is on **Pipeline A**, which handles sarcasm translation in two steps:

1. **Sarcasm Interpretation**:
   - Use Seq2Seq models to convert sarcastic English tweets into **honest English**.
   - Hypothesis: Transformer models (like **BERT**) will outperform RNN-based approaches (e.g., Peled & Reichart, 2017) due to their contextual strength.

2. **Telugu Translation**:
   - Translate the honest English interpretation to Telugu using machine translation techniques.

> This ensures the translated message **accurately conveys the true meaning** behind the sarcastic tweet.

**Pipeline B**, on the other hand, attempts a **direct translation** from sarcastic English to honest Telugu. However, we hypothesize that **Pipeline A will perform better** because:
- Telugu is a **low-resource language**
- English-to-English models show **higher contextual performance** than English-to-low-resource translation

This research aims to:
- Improve informal online communication (see *Figure 2*)
- Foster **cross-lingual understanding** in social media
- Provide a **baseline** for solving other open-class NMT challenges

Ultimately, by integrating sarcasm interpretation into machine translation, we aim to:
- Achieve **high-quality low-resource translations**
- Enhance model performance in handling **complex language features**

#### Figure 1: Current Translation Experience  
![Current](https://github.com/user-attachments/assets/cf9e91fa-5c32-40d5-8ef2-cdae12c5391a)

#### Figure 2: Target Translation Experience  
![TargetExperience](https://github.com/user-attachments/assets/2527c9be-19cd-40b3-b3b1-34c415c2dea4)

---

## **2. Background & Related Work**

There has been significant research around **sarcasm interpretation**, but limited work on **sarcasm translation**, especially in the context of *text-to-text translation of memes*.

While notable progress has been made in sarcasm interpretation using **multi-modal models** [(Desai et al., 2022)], translating sarcastic content remains a challenge.

Sarcasm translation is considered an **open-class Neural Machine Translation (NMT)** problem. This is because the **meaning of sarcastic expressions is not compositional**—it doesn’t arise simply from the meanings of individual words. Models that directly translate such expressions often fail to preserve their intended meaning.

A similar challenge is seen in **idiom translation**, which has seen more progress. For instance, **Baziotis et al. (2022)** provide an evaluation and analysis of idioms, offering useful insights for handling open-class translation problems.

> However, their work does **not address low-resource language adaptations**, which is a key focus in our project.

#### **Sarcasm Interpretation: The SIGN Approach**

Our sarcasm interpretation method builds upon the approach by **Peled and Reichart (2017)**, who framed the problem as a **monolingual machine translation** task. Their model, called **SIGN (Sarcasm Sentiment Interpretation Generator)**, focuses on sentiment words that express the opposite of their literal meaning in sarcastic contexts.

Key steps in SIGN:
1. **Clustering Sentiment Words** into *positive* and *negative* based on semantic similarity
2. **Replacing Sentiment Words** with **cluster IDs** in both:
   - Sarcastic source text
   - Honest reference text
3. **Training a phrase-based MT model** on this transformed data
4. **De-clustering the output** at inference time to recover the honest interpretation

They propose 3 de-clustering strategies:
- **SIGN-centroid**: Replace each cluster ID with the sentiment word closest to its centroid in the word embedding space
- **SIGN-context**: Use *point-wise mutual information* with neighboring words to choose replacements
- **SIGN-oracle**: Use human judgment for the best replacement (upper-bound performance)

> While SIGN did not outperform baselines on automatic metrics, **human evaluation** showed SIGN's outputs better captured **intended sentiment**, especially with the context-based method.

#### **English → Telugu Translation Challenges**

Translating English into **Telugu**, a **low-resource language**, is a multifaceted challenge due to:
- Rich **morphological structure**
- High **syntactic diversity**
- Limited high-quality **parallel corpora**

Prior works like **Prasad and Muthukumaran (2013)** and **Ramesh et al. (2023)** have shown progress in **Indian language MT**, but Telugu still presents unique difficulties.

Advanced transformer-based models like:
- **T5 / mT5** [(Raffel et al., 2020)]
- **mBART** [(Tang et al., 2020)]

... have shown exceptional performance across multilingual tasks by capturing long-range dependencies and rich context.

Additionally, **custom tokenizers** tailored for Telugu are necessary for proper evaluation:
- **Sandhi** (morphophonemic changes)
- **Samasa** (compound word formations)

See also: **IndicTrans2** [(Gala et al., 2023)] — a framework for Indian language MT emphasizing **tokenization techniques** and **cultural nuance preservation**.

> This enables evaluation beyond literal correctness—focusing on cultural and contextual alignment as well.
---

## **3. Methodology**

### **3.1 Datasets**

To evaluate our sarcasm interpretation pipelines, we required a dataset containing:

- English sarcastic sentences
- Corresponding **honest Telugu translations**

Since such a high-quality dataset was unavailable, we extended the **Sarcasm SIGN dataset** [(Peled and Reichart, 2017)]:
- Contains **2,993** unique sarcastic tweets
- Each tweet has **5 English interpretations** → total of **14,965** interpretations

#### Dataset Construction Steps:

1. **Initial Translation**:
   - Used **Google Translate API** to generate Telugu interpretations for the 14,965 English ones.

2. **Manual Correction**:
   - Every Telugu sentence was **manually vetted** and corrected by native Telugu-speaking team members.
   - Goal: Improve semantic alignment and idiomatic correctness.

3. **Common Observations During Correction**:
   - Non-alphabetic symbols sometimes mistranslated or returned as Unicode.
   - Lack of native terminology led to:
     - Leaving English terms unchanged (a common practice in Telugu)
     - Transliteration of English terms into Telugu
   - Resolved named entities (e.g., football teams, companies) using Telugu news sources:
     - [Eenadu](https://www.eenadu.net/)
     - [Sakshi](https://www.sakshi.com/)

> ✅ The **corrected translations** were used as the **ground truth** for model training and evaluation.


### **3.2 Experiment Design**

We present **two schemes** to interpret and translate English sarcasm:

#### **Pipeline A: Two-Step Translation**

1. **Interpretation Phase**:
   - Fine-tune a **seq2seq model** to convert English sarcasm → English honest interpretations

2. **Translation Phase**:
   - Fine-tune a **machine translation (MT)** model to translate:
     English honest interpretation → Telugu honest interpretation

#### **Pipeline B: Direct Translation**

- Fine-tune a **translation model** to map:
  English sarcasm → Telugu honest interpretation


We used **HuggingFace pre-trained models**:

- English to English (Interpretation):
  - `google-t5/t5-*`
  - `facebook/bart-*`

- English to Telugu (Translation):
  - `google/mt5-*`
  - `facebook/mbart-*`

#### Figure 3: Two-Pipelines Approach
![Pipelines](https://github.com/user-attachments/assets/dbe8585b-5306-4881-86c3-87dc1623e259)


#### Training Details:

- **Loss-based early stopping** with patience of 5
- Trained for ~15 epochs
- **Hardware**: 3 × NVIDIA A100-SXM4-80GB GPUs
- **Batch size**: 32
- **Validation/Test split**: 20% each

> Models were selected based on **best validation loss**.


### **3.3 Test Design & Metrics**

We used the following **automated metrics**:

- **BLEU**
- **ROUGE** (1, 2, and L variants)
- **PINC** [(Chen and Dolan, 2011)]
  - N-gram dissimilarity metric
  - Originally designed for **paraphrase tasks**

#### Evaluation Strategy:

- All metrics calculated on **test set**
- **Telugu evaluations** used manually corrected interpretations as references
- Used the **pre-trained tokenizers** from the respective models for fair metric computation [(Ramesh et al., 2023)]

#### **Human Evaluation**

We followed the approach of **Desai et al. (2022)**.

- **25 random samples** were selected from the test set.
- **7 evaluators** (linguistic experts aged 20–30) were asked to rate outputs from the best models of each pipeline.

**Metrics Rated**:
- **Adequacy**: Accuracy of interpreting sarcasm
- **Fluency**: Coherency of the Telugu translation

#### Rating Scale:
- Excellent
- Good
- Fair
- Poor

#### Voting Scheme:
- **Majority voting** used to finalize score per example
- **Tie-breaks**:
  - For 2-way ties: select **lower rating**
  - For longer ties: select **median rating**

---

## **4. Results and Analysis**

We present and compare the performance of our fine-tuned models against existing approaches.

### **4.1 English to English Interpretation (Pipeline A)**

In **Table 1**, we compare our fine-tuned models with **SIGN** [(Peled and Reichart, 2017)].

| **Model**       | **BLEU** | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** | **PINC** |
|-----------------|----------|-------------|-------------|-------------|----------|
| SIGN ‡          | 66.96    | 70.34       | 42.81       | 69.98       | 47.11    |
| T5-base †       | 84.34    | 87.89       | 80.90       | 87.37       | 15.97    |
| T5-large †      | 85.29    | 89.28       | 82.83       | 88.95       | 13.83    |
| BART-large †    | **86.32**| 86.40       | 80.73       | 86.21       | **11.06**|

> Our models achieve **higher BLEU and ROUGE** than SIGN, indicating accurate interpretations.
> Lower PINC suggests smaller surface-level changes, which is expected when sarcasm is subtle.


### **4.2 English to Telugu Translation (Pipeline A & B)**

In **Table 2**, we report BLEU and ROUGE scores for both pipelines.

![Screenshot 2025-03-30 005600](https://github.com/user-attachments/assets/96d74fe4-c514-48f0-a057-ac28fcd95578)

> **Pipeline A** clearly outperforms direct translation (**Pipeline B**), especially in **BLEU score** (35.80 vs 31.69).


### **4.3 Human Evaluation Results**

| **Pipeline** | **Adequacy (avg)** | **Fluency (avg)** |
|--------------|--------------------|-------------------|
| A            | **3.8**            | **3.88**          |
| B            | 3.2                | 3.04              |

> Pipeline A again outperforms in both **interpretation accuracy** and **fluency**.


### **Figure 4: Human Evaluation Samples**
![Screenshot 2024-04-22 at 15 05 30](https://github.com/user-attachments/assets/484f81a4-13c3-41a6-b7b7-35a0bcbfb7d7)



Two **high-rated** and two **low-rated** samples are shown, illustrating how **sentence length** and **punctuation** affected translation quality.

### **Observations:**
- Shorter sentences tend to yield **better results**
- Removing punctuation led to **misinterpreted sarcasm**, affecting translations

---

## **5. Conclusion and Future Work**

In this paper, we explored **two approaches** for achieving accurate **sarcasm interpretation and translation** from **English to Telugu**:

### ✅ **Pipeline A (Two-Stage Approach)**  
1. English sarcasm → English honest interpretation  
2. English honest → Telugu honest translation  

### ✅ **Pipeline B (Direct Translation)**  
- English sarcasm → Telugu honest interpretation  

To effectively fine-tune our models, we **manually curated a Telugu dataset** by correcting Google Translate outputs, ensuring semantic and contextual alignment.  

We evaluated model performance using both:
- **Automatic metrics** (BLEU, ROUGE, PINC)  
- **Human evaluations** (Adequacy, Fluency)


### 🔍 Key Findings:
- **Pipeline A outperformed Pipeline B** on all evaluation metrics.
- **Transformer-based models** (like T5, mBART) significantly improved results.
- Incorporating **sarcasm interpretation as a preprocessing step** enhanced translation quality for **low-resource languages**.
- Human evaluations confirmed the superiority of **Pipeline A** in **fluency and adequacy**.


### 🔭 Future Work:

While this research focused on translating **literal meaning** into Telugu, future work can aim to:

- Enable **direct translation of sarcastic intent** into Telugu or other languages
- Expand the dataset to include **sarcastic phrases**

---

## **References**

- Baziotis, C., Mathur, P., & Hasler, E. (2022). *Automatic evaluation and analysis of idioms in neural machine translation*. [arXiv:2210.04545](http://arxiv.org/abs/2210.04545)

- Chen, D., & Dolan, W. B. (2011). *Collecting highly parallel data for paraphrase evaluation*. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, pp. 190–200.

- Desai, P., Chakraborty, T., & Akhtar, M. S. (2022). *Nice perfume. how long did you marinate in it? multimodal sarcasm explanation*. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 36, 10563–10571.

- Gala, J., Chitale, P. A., Raghavan, A. K., Doddapaneni, S., Gumma, V., Kumar, A., Nawale, J., Sujatha, A., Puduppully, R., Raghavan, V., et al. (2023). *IndicTrans2: Towards high-quality and accessible machine translation models for all 22 scheduled Indian languages*. [arXiv:2305.16307](https://arxiv.org/abs/2305.16307)

- Peled, L., & Reichart, R. (2017). *Sarcasm SIGN: Interpreting sarcasm with sentiment-based monolingual machine translation*. [arXiv:1704.06836](https://arxiv.org/abs/1704.06836)

- Prasad, T. V., & Muthukumaran, G. M. (2013). *Telugu to English translation using direct machine translation approach*. *International Journal of Science and Engineering Investigations*, 2(12), 25–32.

- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). *Exploring the limits of transfer learning with a unified text-to-text transformer*. *Journal of Machine Learning Research*, 21(140), 1–67. [Link](http://jmlr.org/papers/v21/20-074.html)

- Ramesh, G., Doddapaneni, S., Bheemaraj, A., Jobanputra, M., Raghavan, A. K., Sharma, A., Sahoo, S., Diddee, H., J, M., Kakwani, D., et al. (2023). *Samanantar: The largest publicly available parallel corpora collection for 11 Indic languages*.

- Tang, Y., Tran, C., Li, X., Chen, P. J., Goyal, N., Chaudhary, V., Gu, J., & Fan, A. (2020). *Multilingual translation with extensible multilingual pretraining and finetuning*. [arXiv:2008.00401](http://arxiv.org/abs/2008.00401)

---

## Files
- [Download the PDF Version](https://github.com/user-attachments/files/19524329/Sarcasm-Aware.Translation.pdf)

