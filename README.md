# LCPSG
Lack Content Of Punctuation Sentence Generation

This paper mainly uses the pre training language model BART of text generation to generate the missing shared components of input text under the premise of removing the assumption of conditional independence of the first and last positions of missing components; The type prediction multi task learning mechanism of sharing mode is added to improve the model's ability to learn the grammar rules of clause complex, so as to improve the model's performance; In the process of hierarchical header structure analysis, in addition to learning and forecasting the direct headers, it can also learn and predict the indirect headers nested in the hierarchical structure.

# Version
pytorch1.7
transformer4.3
