# QSAR Modeling, Similarity Search, and Virtual Screening to Identify AChE Inhibitors for Alzheimer's Disease

## 📚 **MBA em Inteligência Artificial e Big Data**  
**Universidade de São Paulo (USP)**  
**Instituto de Ciências Matemáticas e de Computação (ICMC)**  

### **Autores:**  
- **Leandro Pedrosa** | leandropedrosalp@gmail.com  
- **Tatiane Nogueira Rios** | tatianenogueira@gmail.com  

---

## 📄 **Resumo**  
Este estudo explora o uso de técnicas de *Machine Learning* e *Deep Learning* para *screening* virtual de potenciais inibidores da acetilcolinesterase (**AChE**) voltados ao tratamento da **Doença de Alzheimer**. Integrando **QSAR Modeling** com descritores moleculares avançados (Morgan, RDKit e SiRMS), os modelos demonstraram alta acurácia e eficiência.

Modelos utilizados incluem:  
- **Support Vector Machine (SVM)**  
- **Random Forest (RF)**  
- **Multilayer Perceptron (MLP)**  
- **TensorFlow**  

O estudo identificou **37 compostos promissores**, selecionados via *consensus modeling* e *similarity search* utilizando o coeficiente **Tanimoto**. Estes compostos apresentaram mais de **50% de similaridade** com moléculas referência, como a **Tacrina**, evidenciando a robustez da abordagem.

---

## 🧠 **Contexto e Motivação**  
A **Doença de Alzheimer** é uma condição neurodegenerativa caracterizada por declínio cognitivo e perda de memória. Inibidores de AChE aumentam os níveis de acetilcolina no cérebro, melhorando a transmissão neuronal e aliviando sintomas cognitivos.  

Métodos tradicionais de descoberta de fármacos são **custosos e demorados**. Este estudo propõe uma abordagem **computacional e eficiente**, combinando modelos QSAR e técnicas de inteligência artificial para acelerar o processo.

---

## 🎯 **Objetivos**  
1. Desenvolver **modelos QSAR** utilizando técnicas de Machine Learning e Deep Learning.  
2. Identificar potenciais **inibidores de AChE** via **screening virtual**.  
3. Integrar *consensus modeling* e buscas de similaridade para selecionar compostos com maior confiabilidade.

---

## 🛠️ **Metodologia**  

### **1. Preparação de Dados**  
- Definição do alvo (AChE).  
- Organização de um dataset contendo **8.832 compostos** da base **ChemBL**.  
- Cálculo de descritores moleculares:  
  - **Morgan Fingerprints**  
  - **RDKit**  
  - **SiRMS**  

### **2. Construção dos Modelos QSAR**  
- Algoritmos: **SVM, RF, MLP, TensorFlow**.  
- **Validação cruzada** e ajuste de hiperparâmetros com **RandomizedSearchCV** e **Keras Tuner**.  

### **3. Validação dos Modelos**  
- Métricas de avaliação: **acurácia**, **F1-Score**, **sensibilidade** e **especificidade**.  
- Teste de permutação para validação estatística.  
- Aplicação do **Applicability Domain (AD)** para aumentar a confiabilidade.

### **4. Screening Virtual**  
- Aplicação dos modelos treinados em **101.097 compostos** (base **PubChem**).  
- *Consensus modeling* para integrar resultados.  
- Busca de similaridade com coeficiente **Tanimoto** para priorização dos compostos.

---

## 📊 **Resultados**  
- **Morgan Descriptors:** Melhor resultado com SVM (acurácia 0,87).  
- **RDKit Descriptors:** MLP alcançou acurácia de **0,90**.  
- **SiRMS Descriptors:** RF e SVM com acurácia de **0,91**.  

### **Resultados do Screening Virtual:**  
- **6.455 hits** identificados com descritores Morgan.  
- **3.773 hits** identificados com RDKit.  
- **5.837 compostos** priorizados via similaridade Tanimoto > 50%.  

Os modelos demonstraram desempenho robusto, com ganhos médios de **20-25%** ao aplicar o conceito de **AD**.

---

## 🔎 **Conclusão**  
A combinação de **QSAR modeling**, **screening virtual** e técnicas de **Machine Learning** representa uma abordagem eficiente para a descoberta de **inibidores de AChE**, com potencial aplicação em fármacos para **Alzheimer**.  

**Destaques:**  
- Identificação de **37 compostos promissores**.  
- Validação robusta com **permutation tests** e **consensus modeling**.  
- Eficiência e escalabilidade para aplicações em larga escala.

---

## 🚀 **Perspectivas Futuras**  
- Refinamento de técnicas de *consensus modeling*.  
- Validação experimental dos compostos identificados.  
- Exploração de novos descritores moleculares e métodos híbridos.  
- Aplicação de **Deep Learning Explicável** para melhor compreensão dos mecanismos moleculares.

---

## 📌 **Referências Principais**  
1. **ChemBL Database**  
2. Morgan Fingerprints, RDKit, SiRMS  
3. Machine Learning Models: SVM, RF, MLP, TensorFlow  
4. Tanimoto Similarity Coefficient  

---

## 🧩 **Tecnologias Utilizadas**  
- **Python**  
- **Scikit-learn**  
- **TensorFlow/Keras**  
- **RDKit**  
- **PubChem e ChemBL Databases**
