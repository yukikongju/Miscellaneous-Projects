-- cifas_headings.lua
-- Filtre Lua Pandoc : remplace les headings ##–###### par du LaTeX custom.
-- Auto-injecte les packages nécessaires dans le préambule.
--
-- Compatible avec ou sans template personnalisé.
-- Usage : pandoc ... --lua-filter cifas_headings.lua
-- ---------------------------------------------------------------------------

local PREAMBLE = [[
\usepackage{xcolor}
\providecolor{cifasGris}{RGB}{90,90,90}
\usepackage[normalem]{ulem}
\usepackage[framemethod=default]{mdframed}
\usepackage{microtype}
\usepackage{needspace}
\ifx\cifasquote\undefined
\newmdenv[
  topline=false, bottomline=false, rightline=false,
  linecolor=cifasGris, linewidth=2pt,
  skipabove=4pt, skipbelow=4pt,
  innerleftmargin=10pt, innerrightmargin=0pt,
  innertopmargin=2pt, innerbottommargin=2pt,
]{cifasquote}
\fi
% Filet horizontal fin jusqu'à la marge
\providecommand{\cifasrulefill}{%
  \leavevmode\leaders\hrule height 0.4pt\hfill\kern0pt}
]]

function Meta(m)
    if FORMAT ~= "latex" then return nil end
    local hi = m["header-includes"]
    if not hi then hi = pandoc.MetaList({}) end
    table.insert(hi, pandoc.MetaBlocks({ pandoc.RawBlock("latex", PREAMBLE) }))
    m["header-includes"] = hi
    return m
end

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

local function to_latex(inlines)
    local doc = pandoc.Pandoc({ pandoc.Para(inlines) })
    local raw = pandoc.write(doc, "latex")
    return raw:match("^%s*(.-)%s*$")
end

local function to_plain(inlines)
    return pandoc.utils.stringify(inlines)
end

local function latex_escape(s)
    s = s:gsub("\\", "\\textbackslash{}")
    s = s:gsub("&", "\\&")
    s = s:gsub("%%", "\\%%")
    s = s:gsub("#", "\\#")
    s = s:gsub("%$", "\\$")
    s = s:gsub("_", "\\_")
    s = s:gsub("{", "\\{")
    s = s:gsub("}", "\\}")
    return s
end

-- Split author paragraph into a 2-column tabular.
local function format_author_block(el)
    local author_lines = {}
    local current = {}
    for _, inline in ipairs(el.content) do
        if inline.t == "LineBreak" then
            if #current > 0 then
                table.insert(author_lines, current)
                current = {}
            end
        else
            table.insert(current, inline)
        end
    end
    if #current > 0 then table.insert(author_lines, current) end

    local authors = {}
    local em_dash = "\xe2\x80\x94"
    local en_dash = "\xe2\x80\x93"
    for _, line_inlines in ipairs(author_lines) do
        local plain = pandoc.utils.stringify(line_inlines)
        local name, inst
        local pos = plain:find(" " .. em_dash .. " ", 1, true)
        if pos then
            name = plain:sub(1, pos - 1)
            inst = plain:sub(pos + 1 + #em_dash + 1)
        else
            pos = plain:find(" " .. en_dash .. " ", 1, true)
            if pos then
                name = plain:sub(1, pos - 1)
                inst = plain:sub(pos + 1 + #en_dash + 1)
            else
                name = plain; inst = ""
            end
        end
        table.insert(authors, { name = latex_escape(name), inst = latex_escape(inst) })
    end

    local rows = {}
    local i = 1
    while i <= #authors do
        local a1 = authors[i]
        local a2 = (i + 1 <= #authors) and authors[i + 1] or nil
        local n2 = a2 and ("\\textbf{" .. a2.name .. "}") or ""
        local i2 = a2 and ("\\textit{\\small " .. a2.inst .. "}") or ""
        table.insert(rows, "\\textbf{" .. a1.name .. "} & " .. n2 .. " \\\\[1pt]")
        table.insert(rows, "\\textit{\\small " .. a1.inst .. "} & " .. i2 .. " \\\\[8pt]")
        i = i + 2
    end
    if #rows > 0 then
        rows[#rows] = rows[#rows]:gsub("%[8pt%]$", "")
    end

    return pandoc.RawBlock("latex", table.concat({
        "\\vspace{4pt}",
        "\\noindent\\begin{tabular}{@{}p{0.46\\linewidth}@{\\hspace{0.06\\linewidth}}p{0.46\\linewidth}@{}}",
        table.concat(rows, "\n"),
        "\\end{tabular}",
        "\\vspace{4pt}",
    }, "\n"))
end

-- ---------------------------------------------------------------------------
-- State : détecter le paragraphe qui suit --- ou ###### (= bloc auteurs)
--         + en-tête courant (jour / bloc)
-- ---------------------------------------------------------------------------
local after_rule  = false
local after_h6    = false
local current_day = ""

local CONF_HEADER = "CONGRÈS INTERNATIONAL FRANCOPHONE SUR L'AGRESSION SEXUELLE 2026"

local function sentence_case(s)
    local low = s:lower()
    return low:sub(1, 1):upper() .. low:sub(2)
end

-- ---------------------------------------------------------------------------
-- Filtre : headings
-- ---------------------------------------------------------------------------

function Header(el)
    if FORMAT ~= "latex" then return nil end

    local lvl   = el.level
    local text  = to_latex(el.content)
    local plain = to_plain(el.content)

    -- ── ## → jour : séparateur gris + caps trackés + filet à droite ────────
    if lvl == 2 then
        current_day = sentence_case(plain)
        return pandoc.RawBlock("latex", table.concat({
            -- "\\markright{" .. CONF_HEADER .. " | " .. current_day .. " | " .. event_label .."}",
            "\\markright{" .. CONF_HEADER .. " | " .. current_day .. "}",
            "\\phantomsection",
            "\\addcontentsline{toc}{subsection}{" .. plain .. "}",
            "\\vspace{20pt}",
            "{\\color{cifasGris!25}\\hrule height 0.4pt}",
            "\\vspace{10pt}",
            "\\noindent{\\small\\sffamily\\bfseries\\textls[200]{" .. text .. "}}\\enskip\\cifasrulefill",
            "\\par\\vspace{10pt}",
        }, "\n"))

        -- ── ### → BLOC ou événement avec heure ─────────────────────────────────
    elseif lvl == 3 then
        local toc = "\\phantomsection\n\\addcontentsline{toc}{subsubsection}{" .. plain .. "}"
        local em_dash = "\xe2\x80\x94"
        local sep = " " .. em_dash .. " "

        -- Calcul du label pour l'en-tête courant
        local event_label
        if plain:match("^BLOC%s") then
            local bloc_p = plain:match("^(BLOC%s+[%d%.]+)")
            event_label = sentence_case(bloc_p or plain)
        else
            local sp = plain:find(sep, 1, true)
            event_label = sentence_case(sp and plain:sub(1, sp - 1) or plain)
        end
        local mark_cmd = "\\markright{" .. CONF_HEADER .. " | " .. current_day .. "}"

        -- Cas BLOC : "BLOC N — hh h mm à hh h mm"
        if plain:match("^BLOC%s") then
            local bloc_p = plain:match("^(BLOC%s+[%d%.]+)")
            local time_p = plain:match(em_dash .. "%s*(.-)%s*$")
            local bloc_l = latex_escape(bloc_p or plain)
            if time_p and time_p ~= "" then
                local time_l = latex_escape(time_p)
                return pandoc.RawBlock("latex", table.concat({
                    mark_cmd,
                    toc,
                    "\\vspace{14pt}",
                    "\\noindent{\\LARGE\\bfseries\\rmfamily " .. bloc_l .. "}\\enskip" ..
                    "\\raisebox{4pt}{\\setlength{\\fboxsep}{2pt}\\setlength{\\fboxrule}{0.5pt}" ..
                    "\\fbox{\\footnotesize\\color{cifasGris} " .. time_l .. " }}" ..
                    "\\enskip\\leaders\\hrule height 4.4pt depth -4pt\\hfill\\kern0pt",
                    "\\par\\vspace{8pt}",
                }, "\n"))
            else
                return pandoc.RawBlock("latex", table.concat({
                    mark_cmd,
                    toc,
                    "\\vspace{14pt}",
                    "\\noindent{\\LARGE\\bfseries\\rmfamily " .. bloc_l .. "}\\enskip\\cifasrulefill",
                    "\\par\\vspace{8pt}",
                }, "\n"))
            end
        end

        -- Cas événement : "NOM — heure" → heure en petit gris au-dessus, nom en grand gras
        local sep_pos = plain:find(sep, 1, true)
        if sep_pos then
            local name_p = plain:sub(1, sep_pos - 1)
            local time_p = plain:sub(sep_pos + #sep)
            local name_l = latex_escape(name_p)
            local time_l = latex_escape(time_p)
            return pandoc.RawBlock("latex", table.concat({
                mark_cmd,
                toc,
                "\\vspace{14pt}",
                "\\noindent{\\small\\color{cifasGris}" .. time_l .. "}",
                "\\par\\nobreak\\vspace{2pt}",
                "\\noindent{\\Large\\bfseries\\rmfamily " .. name_l .. "}",
                "\\par\\vspace{6pt}",
            }, "\n"))
        end

        -- Fallback (pas de séparateur) : souligné
        return pandoc.RawBlock("latex", table.concat({
            mark_cmd,
            toc,
            "\\vspace{10pt}",
            "\\noindent{\\fontsize{14}{15}\\selectfont\\bfseries\\sffamily",
            "  \\uline{" .. text .. "}}",
            "\\par\\vspace{5pt}",
        }, "\n"))

        -- ── #### → catégorie (SYMPOSIUMS…) : petites caps grises, sans filet ───
    elseif lvl == 4 then
        return pandoc.RawBlock("latex", table.concat({
            "\\vspace{10pt}",
            "\\noindent{\\footnotesize\\sffamily\\color{cifasGris}\\textls[200]{" .. text .. "}}",
            "\\par\\vspace{4pt}",
        }, "\n"))

        -- ── ##### → salle : « Bloc X.D | Salle — Niveau N » + filet gris ───────
    elseif lvl == 5 then
        local bar_pos = text:find(" | ", 1, true)
        local content
        if bar_pos then
            local left  = text:sub(1, bar_pos - 1)
            local right = text:sub(bar_pos + 3)
            content     = "{\\normalsize\\bfseries " .. left ..
                "}\\enspace\\textbar\\enspace{\\normalsize\\itshape " .. right .. "}"
        else
            content = "{\\normalsize\\bfseries " .. text .. "}"
        end
        return pandoc.RawBlock("latex", table.concat({
            "\\vspace{8pt}",
            "\\noindent" .. content,
            "\\par\\vspace{4pt}",
            "{\\color{cifasGris!40}\\hrule height 0.4pt}",
            "\\vspace{8pt}",
        }, "\n"))

        -- ── ###### → présentation : numéro petit gris + titre \LARGE\bfseries ──
    elseif lvl == 6 then
        local numero = plain:match("^# (%d+)")
        local titre  = text
        if numero then
            titre = text:gsub("^\\#%s*%d+%s*", "", 1)
            if titre == text then
                titre = text:gsub("^#%s*%d+%s*", "", 1)
            end
        end
        after_h6 = true
        local num_block = numero
            and ("\\noindent{\\small\\color{cifasGris}\\# " .. numero .. "}\\par\\nobreak\\vspace{2pt}\n")
            or ""
        return pandoc.RawBlock("latex", table.concat({
            -- "\\needspace{10em}",
            "\\needspace{6em}",
            "\\vspace{12pt}",
            num_block .. "\\noindent{\\LARGE\\bfseries\\rmfamily " .. titre .. "}",
            "\\par\\nobreak\\vspace{4pt}",
        }, "\n"))
    end

    return nil
end

-- ── Règle horizontale ────────────────────────────────────────────────────
function HorizontalRule()
    if FORMAT ~= "latex" then return nil end
    after_rule = true
    return pandoc.RawBlock("latex", "\\vspace{10pt}")
end

-- Détecte un paragraphe auteur : **Nom** — Institution (Strong partiel + tiret cadratin)
local function is_author_para(el)
    if not (el.content[1] and el.content[1].t == "Strong") then return false end
    if #el.content <= 1 then return false end   -- ligne entièrement en gras → pas auteur
    local em_dash = "\xe2\x80\x94"
    local en_dash = "\xe2\x80\x93"
    local rest = ""
    for i = 2, #el.content do
        rest = rest .. pandoc.utils.stringify(el.content[i])
    end
    return rest:sub(1, 1 + #em_dash - 1) == (" " .. em_dash)
        or rest:sub(1, 1 + #en_dash - 1) == (" " .. en_dash)
end

-- ── Paragraphes ──────────────────────────────────────────────────────────
function Para(el)
    if FORMAT ~= "latex" then return nil end

    after_rule = false
    after_h6   = false

    -- Label RÉSUMÉ
    if pandoc.utils.stringify(el.content) == "RÉSUMÉ" then
        return pandoc.RawBlock("latex",
            "\\vspace{6pt}\\noindent{\\footnotesize\\sffamily\\color{cifasGris}\\textls[200]{RÉSUMÉ}}\\par\\vspace{2pt}")
    end

    -- Bloc auteurs : **Nom** — Institution (détection par contenu, sans dépendance de position)
    if is_author_para(el) then
        return format_author_block(el)
    end

    -- Tous les autres paragraphes : noindent pour aligner sur la marge du texte
    local latex = to_latex(el.content)
    if latex ~= "" then
        return pandoc.RawBlock("latex", "\\noindent " .. latex .. "\n")
    end

    return nil
end

-- ── Blockquote → filet gauche gris (env. cifasquote) ─────────────────────
function BlockQuote(el)
    if FORMAT ~= "latex" then return nil end
    local inner = pandoc.write(pandoc.Pandoc(el.content), "latex"):match("^%s*(.-)%s*$")
    return pandoc.RawBlock("latex", table.concat({
        "\\noindent\\begin{cifasquote}",
        "\\setlength{\\parindent}{0pt}\\rmfamily\\normalsize",
        inner,
        "\\end{cifasquote}",
    }, "\n"))
end
