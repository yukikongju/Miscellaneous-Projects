-- cifas_headings.lua
-- Filtre Lua Pandoc : remplace les headings ##–###### par du LaTeX custom.
-- Auto-injecte les packages nécessaires (xcolor, mdframed) dans le préambule.
--
-- Compatible avec ou sans template personnalisé.
-- Usage : pandoc ... --lua-filter cifas_headings.lua
-- ---------------------------------------------------------------------------

local PREAMBLE = [[
\usepackage{xcolor}
\providecolor{cifasGris}{RGB}{90,90,90}
\usepackage[normalem]{ulem}
\usepackage[framemethod=default]{mdframed}
\ifx\cifasquote\undefined
\newmdenv[
  topline=false, bottomline=false, rightline=false,
  linecolor=cifasGris, linewidth=2pt,
  skipabove=4pt, skipbelow=4pt,
  innerleftmargin=10pt, innerrightmargin=0pt,
  innertopmargin=2pt, innerbottommargin=2pt,
]{cifasquote}
\fi
]]

function Meta(m)
    if FORMAT ~= "latex" then return nil end
    local hi = m["header-includes"]
    if not hi then hi = pandoc.MetaList({}) end
    table.insert(hi, pandoc.MetaBlocks({ pandoc.RawBlock("latex", PREAMBLE) }))
    m["header-includes"] = hi
    return m
end

-- Convertit des Inlines Pandoc en chaîne LaTeX (gère l'échappement).
local function to_latex(inlines)
    local doc = pandoc.Pandoc({ pandoc.Para(inlines) })
    local raw = pandoc.write(doc, "latex")
    -- Supprimer les newlines en fin de paragraphe produits par pandoc.write
    return raw:match("^%s*(.-)%s*$")
end

-- Version texte brut (pour les entrées de table des matières).
local function to_plain(inlines)
    return pandoc.utils.stringify(inlines)
end

-- ---------------------------------------------------------------------------
-- Filtre principal
-- ---------------------------------------------------------------------------

function Header(el)
    -- Ne s'applique qu'à la sortie LaTeX / PDF
    if FORMAT ~= "latex" then return nil end

    local lvl   = el.level
    local text  = to_latex(el.content)
    local plain = to_plain(el.content)

    -- ── ## → rectangle encadré ──────────────────────────────────────────────
    if lvl == 2 then
        return pandoc.RawBlock("latex", table.concat({
            -- Entrée TOC (niveau subsection)
            "\\phantomsection",
            "\\addcontentsline{toc}{subsection}{" .. plain .. "}",
            -- Visuel : fbox pleine largeur
            "\\vspace{14pt}",
            "\\noindent",
            "{\\setlength{\\fboxsep}{6pt}%",
            "\\fbox{\\parbox{\\dimexpr\\linewidth-2\\fboxsep-2\\fboxrule\\relax}{%",
            "  \\large\\sffamily\\bfseries ", text,
            "}}}",
            "\\par\\vspace{8pt}",
        }, "\n"))

        -- ── ### → souligné 14 pt gras ───────────────────────────────────────────
    elseif lvl == 3 then
        return pandoc.RawBlock("latex", table.concat({
            "\\phantomsection",
            "\\addcontentsline{toc}{subsubsection}{" .. plain .. "}",
            "\\vspace{10pt}",
            "\\noindent{\\fontsize{14}{15}\\selectfont\\bfseries\\sffamily",
            "  \\uline{" .. text .. "}}",
            "\\par\\vspace{5pt}",
        }, "\n"))

        -- ── #### → Large gras (salle / bloc de présentation) ───────────────────
    elseif lvl == 4 then
        return pandoc.RawBlock("latex", table.concat({
            "\\vspace{8pt}",
            "\\noindent{\\Large\\bfseries\\rmfamily " .. text .. "}",
            "\\par\\vspace{4pt}",
        }, "\n"))

        -- ── ##### → idem ####, mais \nobreak pour coller au contenu ────────────
    elseif lvl == 5 then
        return pandoc.RawBlock("latex", table.concat({
            "\\vspace{8pt}",
            "\\noindent{\\Large\\bfseries\\rmfamily " .. text .. "}",
            "\\par\\nobreak\\vspace{3pt}",
        }, "\n"))

        -- ── ###### → titre de présentation (#N) : large gras + collé au contenu ─
    elseif lvl == 6 then
        return pandoc.RawBlock("latex", table.concat({
            "\\vspace{8pt}",
            "\\noindent{\\large\\bfseries\\rmfamily " .. text .. "}",
            "\\par\\nobreak\\vspace{3pt}",
        }, "\n"))
    end

    -- Autres niveaux : comportement Pandoc par défaut
    return nil
end

-- ── Règle horizontale → séparateur gris fin ─────────────────────────────
function HorizontalRule()
    if FORMAT ~= "latex" then return nil end
    return pandoc.RawBlock("latex",
        "\\vspace{4pt}{\\color{cifasGris}\\hrule height 0.4pt}\\vspace{8pt}")
end

-- ── Paragraphe "RÉSUMÉ" → label sffamily gris ───────────────────────────
function Para(el)
    if FORMAT ~= "latex" then return nil end
    if pandoc.utils.stringify(el.content) == "RÉSUMÉ" then
        return pandoc.RawBlock("latex",
            "\\vspace{4pt}{\\small\\sffamily\\color{cifasGris}RÉSUMÉ}\\par\\vspace{2pt}")
    end
    return nil
end

-- ── Blockquote → corps avec filet gauche gris (env. cifasquote) ──────────
function BlockQuote(el)
    if FORMAT ~= "latex" then return nil end
    local inner = pandoc.write(pandoc.Pandoc(el.content), "latex"):match("^%s*(.-)%s*$")
    return pandoc.RawBlock("latex", table.concat({
        "\\begin{cifasquote}",
        "\\rmfamily\\normalsize",
        inner,
        "\\end{cifasquote}",
    }, "\n"))
end
