\noindent\textbf{Q:} The core mechanism (FFT + MLP for amplitude/phase scaling) is relatively simplistic. Similar frequency-domain adaptation strategies exist in other domains (e.g., image processing), and the adaptation to time-series feels incremental rather than groundbreaking. The related work section acknowledges MAPU [25] uses similar masking, and the frequency extension appears to be a minor modification.

\vspace{2mm}
\noindent\textbf{A:} We sincerely thank the reviewer for the comment. While MAPU provides a solid foundation for Time-Series SFDA, our Frequency Adaptation Layer (FAL) represents a fundamental paradigm shift rather than a minor extension. Specifically, our contributions are threefold:

\begin{itemize}
    \item \textbf{Novel Adaptation Paradigm:} Standard SFDA methods (including MAPU) continuously update the feature extractor or classifier to align latent features. In contrast, our method \textit{freezes} the entire pre-trained model and achieves alignment directly at the \textbf{signal level} via frequency-domain modulation.
    
    \item \textbf{Extreme Parameter Efficiency:} By freezing the source model, adaptation requires optimizing only two highly lightweight MLPs. This drastically reduces the number of trainable parameters, mitigating computational overhead and the risk of catastrophic forgetting.
    
    \item \textbf{Universality and Elegance:} We view the structural simplicity of FAL as a major strength. It elegantly solves complex domain shifts without convoluted over-engineering. Furthermore, FAL is highly universal—serving as a plug-and-play module that can be seamlessly integrated into other SFDA architectures to provide new insights.
\end{itemize}