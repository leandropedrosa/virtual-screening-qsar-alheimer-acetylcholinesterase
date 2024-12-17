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

## 📄 **Abstract (English)**  

This study investigates the use of **machine learning** and **deep learning** techniques for virtual screening to identify potential **acetylcholinesterase (AChE) inhibitors** as a treatment for **Alzheimer's disease**. By integrating **Quantitative Structure-Activity Relationship (QSAR) modeling** with diverse molecular descriptors—**Morgan fingerprints**, **RDKit**, and **SiRMS**—the study achieved high predictive accuracy and efficiency. Machine learning models, including **Support Vector Machine (SVM)**, **Random Forest (RF)**, **Multilayer Perceptron (MLP)**, and **TensorFlow**, were employed to classify compounds as active or inactive. Consensus modeling combined with similarity searches using the **Tanimoto coefficient** identified **37 promising compounds** with structural similarity greater than 50% to reference molecules such as **tacrine**. These findings underscore the potential of computational approaches to accelerate drug discovery processes, offering an efficient and cost-effective strategy for identifying novel therapeutic candidates to treat neurodegenerative diseases like Alzheimer's.  

---

## 📄 **Resumen (Español)**  

Este estudio investiga el uso de técnicas de **aprendizaje automático** y **aprendizaje profundo** para realizar *screening virtual* y así identificar posibles **inhibidores de la acetilcolinesterasa (AChE)** como tratamiento para la **enfermedad de Alzheimer**. Integrando el **modelado QSAR (Relación Cuantitativa Estructura-Actividad)** con descriptores moleculares diversos—**Morgan fingerprints**, **RDKit** y **SiRMS**—el estudio logró alta precisión predictiva y eficiencia. Se aplicaron modelos de aprendizaje automático como **Support Vector Machine (SVM)**, **Random Forest (RF)**, **Multilayer Perceptron (MLP)** y **TensorFlow** para clasificar compuestos como activos o inactivos. La combinación de *consensus modeling* y búsquedas de similitud usando el **coeficiente de Tanimoto** permitió identificar **37 compuestos prometedores** con una similitud estructural superior al 50% respecto a moléculas de referencia como la **tacrina**. Estos resultados destacan el potencial de los enfoques computacionales para acelerar el descubrimiento de fármacos, proporcionando una estrategia eficiente y rentable para encontrar nuevos candidatos terapéuticos para tratar enfermedades neurodegenerativas como el Alzheimer.  

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

## 🔗 **Link para a Monografia**  
O documento completo da pesquisa pode ser acessado através do link:  
[**Leandro Pedrosa - Monografia**](https://bdta.abcd.usp.br/directbitstream/a9d4d9ea-7691-4462-ac64-ce6bcbaf2d36/Leandro%20Pedrosa.pdf)

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

---

## 📚 **Como citar**  

*A data do acesso citação deve ser adicionada no dia do acesso e pode não estar totalmente de acordo com as normas.*  

### **ABNT**  
PEDROSA, Leandro. Modelagem QSAR (Relação Quantitativa Estrutura-Atividade), busca por similaridade e triagem virtual para a identificação de inibidores de Acetilcolinesterase (AChE) para a doença de Alzheimer. 2023. Trabalho de Conclusão de Curso (MBA) – Instituto de Ciências Matemáticas e de Computação, Universidade de São Paulo, São Carlos, 2023. Disponível em: [https://bdta.abcd.usp.br/directbitstream/a9d4d9ea-7691-4462-ac64-ce6bcbaf2d36/Leandro%20Pedrosa.pdf](https://bdta.abcd.usp.br/directbitstream/a9d4d9ea-7691-4462-ac64-ce6bcbaf2d36/Leandro%20Pedrosa.pdf). Acesso em: 17 dez. 2024.  

### **APA**  
Pedrosa, L. (2023). *Modelagem QSAR (Relação Quantitativa Estrutura-Atividade), busca por similaridade e triagem virtual para a identificação de inibidores de Acetilcolinesterase (AChE) para a doença de Alzheimer* (Trabalho de Conclusão de Curso (MBA). Instituto de Ciências Matemáticas e de Computação, Universidade de São Paulo, São Carlos. Recuperado de [https://bdta.abcd.usp.br/directbitstream/a9d4d9ea-7691-4462-ac64-ce6bcbaf2d36/Leandro%20Pedrosa.pdf](https://bdta.abcd.usp.br/directbitstream/a9d4d9ea-7691-4462-ac64-ce6bcbaf2d36/Leandro%20Pedrosa.pdf)  

### **NLM**  
Pedrosa L. *Modelagem QSAR (Relação Quantitativa Estrutura-Atividade), busca por similaridade e triagem virtual para a identificação de inibidores de Acetilcolinesterase (AChE) para a doença de Alzheimer* [Internet]. 2023; [citado 2024 dez. 17]. Available from: [https://bdta.abcd.usp.br/directbitstream/a9d4d9ea-7691-4462-ac64-ce6bcbaf2d36/Leandro%20Pedrosa.pdf](https://bdta.abcd.usp.br/directbitstream/a9d4d9ea-7691-4462-ac64-ce6bcbaf2d36/Leandro%20Pedrosa.pdf)  

### **Vancouver**  
Pedrosa L. *Modelagem QSAR (Relação Quantitativa Estrutura-Atividade), busca por similaridade e triagem virtual para a identificação de inibidores de Acetilcolinesterase (AChE) para a doença de Alzheimer* [Internet]. 2023; [citado 2024 dez. 17]. Available from: [https://bdta.abcd.usp.br/directbitstream/a9d4d9ea-7691-4462-ac64-ce6bcbaf2d36/Leandro%20Pedrosa.pdf](https://bdta.abcd.usp.br/directbitstream/a9d4d9ea-7691-4462-ac64-ce6bcbaf2d36/Leandro%20Pedrosa.pdf)  

### **BibTeX**  
```bibtex
@misc{miscef9ada78,
  title   = {Modelagem QSAR (Relação Quantitativa Estrutura-Atividade), busca por similaridade e triagem virtual para a identificação de inibidores de Acetilcolinesterase (AChE) para a doença de Alzheimer},
  author  = {Pedrosa, Leandro and Rios, Tatiane Nogueira},
  year    = {2023}
}
```

### **Registro BDTA USP**  
A monografia pode ser acessada no **BDTA USP** através do link:  
[https://bdta.abcd.usp.br/item/003190344](https://bdta.abcd.usp.br/item/003190344)  
