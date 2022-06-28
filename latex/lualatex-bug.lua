

-- \directlua{dofile('lualatex-bug.lua')}
-- \newcommand{\trigtable}{\directlua{trigtable()}}
-- \begin{tabular}{rcccc}
--     \hline
--      & $x$ & $\sin(x)$ & $\cos(x)$ & $\tan(x)$ \\
--     \hline
--     \trigtable
--     \hline
-- \end{tabular}

function trigtable ()
    for t=0, 45, 3 do
        x=math.rad(t)
        tex.print(string.format([[%2d$^{\circ}$ & %1.9f & %1.9f & %1.9f & %1.9f \\]],
                                t, x, math.sin(x), math.cos(x), math.tan(x)))
    end
end
