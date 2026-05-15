BAN_PATTERNS = [

    # =====================================================
    # IMAGE / MEDIA EXTRACTION
    # =====================================================

    r"\[IMAGE:.*?\]",
    r"page_\d+_img_\d+",
    r"image extracted",
    r"embedded image",
    r"figure image",
    r"img_\d+",
    r"\.png",
    r"\.jpeg",
    r"\.jpg",

    # =====================================================
    # PAGE STRUCTURE
    # =====================================================

    r"===== PAGE \d+ =====",
    r"page\s+\d+",
    r"^\d+$",
    r"^\-\s*\d+\s*\-$",
    r"\|\s*page",
    r"chapter\s+\d+",
    r"chapter\s+[ivx]+",
    r"preface",
    r"contents",
    r"table of contents",
    r"summary and outlook",
    r"acknowledgments",
    r"revision history",
    r"index",
    r"appendix",

    # =====================================================
    # OCR GARBAGE
    # =====================================================

    r"[�■□▪▫◆◇]+",
    r"^[^\w\s]{3,}$",
    r"^\W+$",
    r"[|]{2,}",
    r"[.]{5,}",
    r"[\-_]{5,}",
    r"[\*]{3,}",
    r"[\=]{3,}",
    r"[\#]{3,}",
    r"\biii\b",
    r"\biv\b",
    r"\bvii\b",
    r"\bviii\b",
    r"\bix\b",
    r"\bx\b",
    r"\bxi\b",
    r"\bxii\b",

    # =====================================================
    # COPYRIGHT / LEGAL
    # =====================================================

    r"copyright\s*©?",
    r"all rights reserved",
    r"unauthorized reproduction",
    r"published by",
    r"printed in",
    r"permissions@",
    r"terms and conditions",
    r"privacy policy",
    r"license agreement",
    r"without limitation responsibility",
    r"use of the information",
    r"corporate/institutional sales",
    r"oreilly books may be purchased",
    r"this book is here to help you",
    r"fair use",
    r"permission",
    r"trade dress",
    r"registered trademark",

    # =====================================================
    # PUBLISHER / BOOK METADATA
    # =====================================================

    r"isbn",
    r"library of congress",
    r"oreilly",
    r"o’reilly",
    r"safari books online",
    r"prentice hall",
    r"microsoft press",
    r"packt",
    r"apress",
    r"manning",
    r"mcgraw-hill",
    r"editor:",
    r"production editor:",
    r"copyeditor:",
    r"proofreader:",
    r"indexer:",
    r"interior designer:",
    r"cover designer:",
    r"illustrator:",
    r"first edition",
    r"first release",
    r"release details",

    # =====================================================
    # URLS / EMAILS / CONTACTS
    # =====================================================

    r"http\S+",
    r"https\S+",
    r"www\.\S+",
    r"\S+@\S+",
    r"\+?\d[\d\-\s]{7,}",
    r"facebook\.com",
    r"twitter\.com",
    r"youtube\.com",

    # =====================================================
    # WEBSITE / NAVIGATION JUNK
    # =====================================================

    r"click here",
    r"next chapter",
    r"previous page",
    r"home\s+about\s+contact",
    r"subscribe now",
    r"buy premium",
    r"login",
    r"sign up",
    r"cookie policy",
    r"accept cookies",
    r"advertisement",

    # =====================================================
    # HTML / MARKDOWN / XML
    # =====================================================

    r"<[^>]+>",
    r"&nbsp;",
    r"<div>",
    r"<span>",
    r"<script>",
    r"<style>",
    r"```.*?```",
    r"---",
    r"\*\*\*",
    r"###",

    # =====================================================
    # LATEX / FORMULA NOISE
    # =====================================================

    r"\\begin\{.*?\}",
    r"\\end\{.*?\}",
    r"\$\$.*?\$\$",

    # =====================================================
    # WATERMARKS
    # =====================================================

    r"draft",
    r"confidential",
    r"sample copy",

    # =====================================================
    # FIGURE / TABLE CAPTIONS
    # =====================================================

    r"figure\s+\d+",
    r"table\s+\d+",
    r"fig\.\s*\d+",
    r"source:",
    r"historical mentions",

    # =====================================================
    # BROKEN PDF EXTRACTION
    # =====================================================

    r"\b\w+\-\n\w+\b",
    r"\n{2,}",
    r"\s{2,}",

    # =====================================================
    # ENCODING GARBAGE
    # =====================================================

    r"Ã©",
    r"â€™",
    r"ﬁ",
    r"ﬂ",

    # =====================================================
    # VERY LOW INFORMATION
    # =====================================================

    r"^[a-zA-Z]$",
    r"^[0-9]+$",
    r"^[\W_]+$",
    r"^\s*$",

    # =====================================================
    # REFERENCES / CITATIONS
    # =====================================================

    r"\[\d+\]",
    r"et al\.",
    r"ibid",
    r"doi:",
    r"references",
    r"bibliography",

    # =====================================================
    # CSS / JS FRAGMENTS
    # =====================================================

    r"\.class",
    r"function\s*\(",
    r"var\s+\w+",
    r"const\s+\w+",

    # =====================================================
    # JSON / XML ARTIFACTS
    # =====================================================

    r'"\w+"\s*:',
    r"\{.*?\}",

    # =====================================================
    # DIAGRAM OCR JUNK
    # =====================================================

    r"arrow",
    r"node",
    r"diagram",

    # =====================================================
    # GENERIC FILLER SENTENCES
    # =====================================================

    r"this chapter discusses",
    r"in this section",
    r"we will see",
    r"throughout the book",
    r"for more information",

    # =====================================================
    # AI / OCR EXTRACTION ARTIFACTS
    # =====================================================

    r"text extracted by ocr",
    r"scanned by",
    r"converted by",

    # =====================================================
    # PYTHON NOTEBOOK OUTPUT NOISE
    # =====================================================

    r"in\[\d+\]",
    r"out\[\d+\]",
    r">>>",
    r"\[\[\s*\d+",
    r"print\(",

    # =====================================================
    # DATAFRAME DISPLAY NOISE
    # =====================================================

    r"age\s+location\s+name",
    r"display\(",
    r"dataframe",

    # =====================================================
    # BOOK-SPECIFIC REPETITIONS
    # =====================================================

    r"machine learning with python",
    r"machine learning for absolute beginners",
    r"introduction to machine learning",
    r"a guide for data scientists",

    # =====================================================
    # NOTE / TIP / WARNING BOXES
    # =====================================================

    r"this element signifies",
    r"tip or suggestion",
    r"warning or caution",

    # =====================================================
    # MULTI COLUMN PDF DAMAGE
    # =====================================================

    r"\|\s*[a-zA-Z]+\s*\|",

    # =====================================================
    # EMPTY / SYMBOL CHUNKS
    # =====================================================

    r"^[\.\-\_\=\*\#\s]+$",
]