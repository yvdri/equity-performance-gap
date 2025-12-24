# AI Usage

## Code Organization

Since I had started my project before the “Final Project Workshop,” I used ChatGPT to build my project tree structure and then implemented the “Final Project Workshop” to complete my structure. I then worked on my notebooks to explore the models. It was during this phase, with Claude's help, that I discovered versions of Ridge and Lasso with integrated cross-validation (RidgeCV, LassoCV), which guided me through the documentation I needed to apply these different models.

## Learning new things

I already had a good understanding of econometrics from my econometrics course during my bachelor's degree in Neuchâtel. But we hadn't implemented them in Python. This is where certain concepts such as AIC and BIC for comparing linear models come from.

It was Claude who introduced me to RidgeCV and LassoCV. At first, I used normal Ridge and Lasso as in the course, but Claude told me that there were versions with cross-validation already built in. I found that interesting, so I did my research in the sklearn documentation to understand how it works and how to apply it. This allowed me to avoid doing cross-validation manually.

The main reason Claude helped me was with the documentation. Through my exploration, I realized that I lacked understanding of basic libraries and that this was essential for manipulating data. So, in addition to the various libraries included in the course, I used Claude to help me research basic libraries such as matplotlib, sklearn, statsmodels, etc.

## Debugging

When I ran my code and it didn't work, I used Claude to help me understand the errors. This saved me a lot of time compared to searching on Stack Overflow.

## Documentation

Since Claude and ChatGPT helped me with data retrieval, research, and understanding libraries, it was very useful for me to list my sources accurately and quickly.

## Analysis review

All the economic thinking and analytical choices came from my own analysis, such as adding lagged variables because I thought that past values could influence current returns. The choice to add only X1² and X3² also came from my economic reasoning—I thought that the exchange rate and volatility could have nonlinear effects.

I enlisted Claude's help to check whether my assumptions and interpretations of the results were correct and complete, to make sure I wasn't missing anything essential.

## Code review suggestions

Sometimes when my code had bugs, I liked having Claude tell me exactly what was wrong and how to fix it. Some of these corrections were typos or syntax errors. Other times, Claude helped me understand how certain functions worked and how to correct them properly.

## Writing the LaTeX report

Having never used LaTeX before, I used AI (mainly Claude) to help me with the technical aspects of formatting the report:

**Technical LaTeX formatting:**
- Syntax of mathematical equations (`\begin{equation}`, `\frac{}{}`, etc.)
- Cross-references between sections and tables (`\label{}`, `\ref{}`)
- Table formatting (`tabular`, column alignment)
- Inserting and positioning figures (`\includegraphics`, `[H]`, `width=`)
- List structure (`itemize`, `enumerate`)
- Code formatting (`\texttt{}`, `lstlisting`)

The AI essentially served as a “translator” to convert my ideas and text into correct LaTeX syntax, without which I would have spent a lot of time searching through LaTeX documentation for formatting details. I also occasionally used IA for suggesting more precise English phrasings to improve clarity for the readers.

## In summary

Claude and ChatGPT were very useful for learning new things, debugging my code, documentation, and implementation suggestions.