import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = "BertBlocks"
copyright = "2025, CORAL Project Contributors"
author = "CORAL Project Contributors"

version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    #"sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "myst_parser"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "test_*.py"]
python_maximum_signature_line_length = 120

# -- AutoAPI configuration --------------------------------------------------
autoapi_dirs = ["../bertblocks"]
autoapi_type = "python"
autoapi_member_order = "groupwise"
autoapi_keep_files = False
autoapi_ignore = [
    "*/.venv/*",
    "*/site-packages/*",
    "*/__pycache__/*",
    "*/.*",
    "*/test_*.py",
    "./__init__.py",
]

def skip_docstring_attributes(app, what, name, obj, skip, options):
    """
    Skip AutoAPI 'attribute' members if they are already documented
    in the class docstring under 'Attributes:' and thus supplied by napoleon.
    """
    if what == "attribute":
        return True
    return skip  # default behavior


def setup(app):
    app.connect("autoapi-skip-member", skip_docstring_attributes)

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/bertblocks.svg"
html_copy_source = False
html_show_sourcelink = True
html_title = "<release>"

# -- Furo theme configuration ------------------------------------------------
html_theme_options = {
    "sidebar_hide_name": True,
    "top_of_page_buttons": [],
    "source_repository": "https://github.com/coral-nlp/bertblocks/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    # colors based on https://gien.app/pigments/downloads/pigments.css
    "light_css_variables": {
        # Fonts
        "font-stack": "Barlow, Helvetica Neue, sans-serif",
        "font-stack--monospace": "Inconsolata, monospace",
        "font-stack--headings": "Barlow, Helvetica Neue, sans-serif",

        # Base Colors
        "color-foreground-primary": "#3d3b38",  # gray-1000
        "color-foreground-secondary": "#615f5d",  # gray-800
        "color-foreground-muted": "#75726f",  # gray-700
        "color-foreground-border": "#9d9a97",  # gray-500

        "color-background-primary": "#ffffff",  # white
        "color-background-secondary": "#f1f0ef",  # gray-100
        "color-background-hover": "#dddad7",  # gray-200
        "color-background-hover--transparent": "#dddad700",
        "color-background-border": "#c8c4bf",  # gray-300
        "color-background-item": "#b3afab",  # gray-400

        "color-problematic": "#cb4245",  # red-700

        # Announcements
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#f1f0ef",

        # Brand
        "color-brand-primary": "#d9a850",  # yellow-400
        "color-brand-content": "#b77927",  # yellow-600
        "color-brand-visited": "#7e62c0",  # violet-700

        # API documentation
        "color-api-background": "transparent",
        "color-api-background-hover": "#f1f0ef",
        "color-api-overall": "#615f5d",
        "color-api-name": "#cb4245",
        "color-api-pre-name": "#cb4245",
        "color-api-paren": "#615f5d",
        "color-api-keyword": "#3d3b38",

        "color-api-added": "#677851",
        "color-api-added-border": "#89a377",
        "color-api-changed": "#176778",
        "color-api-changed-border": "#4590a9",
        "color-api-deprecated": "#9f6520",
        "color-api-deprecated-border": "#f6e299",
        "color-api-removed": "#cb4245",
        "color-api-removed-border": "#e15953",

        "color-highlight-on-target": "#f9f2c8",

        # Inline code
        "color-inline-code-background": "#f1f0ef",

        # Search highlight
        "color-highlighted-background": "#b7eceb",
        "color-highlighted-text": "#3d3b38",

        # GUI labels
        "color-guilabel-background": "#e2f5f0",
        "color-guilabel-border": "#b7eceb",
        "color-guilabel-text": "#3d3b38",

        # Tables
        "color-table-header-background": "#f1f0ef",
        "color-table-border": "#c8c4bf",

        # Cards
        "color-card-border": "#f1f0ef",
        "color-card-background": "#ffffff",
        "color-card-marginals-background": "#f1f0ef",

        # Header
        "color-header-background": "#ffffff",
        "color-header-border": "#c8c4bf",
        "color-header-text": "#3d3b38",

        # Sidebar
        "color-sidebar-background": "#f1f0ef",
        "color-sidebar-background-border": "#c8c4bf",

        "color-sidebar-brand-text": "#3d3b38",
        "color-sidebar-caption-text": "#75726f",
        "color-sidebar-link-text": "#615f5d",
        "color-sidebar-link-text--top-level": "#d9a850",

        "color-sidebar-item-background": "#f1f0ef",
        "color-sidebar-item-background--current": "#f1f0ef",
        "color-sidebar-item-background--hover": "#dddad7",

        "color-sidebar-item-expander-background": "transparent",
        "color-sidebar-item-expander-background--hover": "#dddad7",

        "color-sidebar-search-text": "#3d3b38",
        "color-sidebar-search-background": "#f1f0ef",
        "color-sidebar-search-background--focus": "#ffffff",
        "color-sidebar-search-border": "#c8c4bf",
        "color-sidebar-search-icon": "#75726f",

        # TOC
        "color-toc-background": "#ffffff",
        "color-toc-title-text": "#75726f",
        "color-toc-item-text": "#615f5d",
        "color-toc-item-text--hover": "#3d3b38",
        "color-toc-item-text--active": "#d9a850",

        # Links
        "color-link": "#b77927",
        "color-link-underline": "#c8c4bf",
        "color-link--hover": "#d9a850",
        "color-link-underline--hover": "#9d9a97",
        "color-link--visited": "#7e62c0",
        "color-link-underline--visited": "#c8c4bf",
        "color-link--visited--hover": "#9f86d0",
        "color-link-underline--visited--hover": "#9d9a97",

        # Admonitions
        "color-admonition-background": "#f1f0ef",
        "color-admonition-title": "#615f5d",
        "color-admonition-title-background": "#dddad7",

        "color-admonition-title--caution": "#b77927",  # yellow-600
        "color-admonition-title-background--caution": "rgba(183, 121, 39, 0.2)",

        "color-admonition-title--warning": "#b77927",  # yellow-600
        "color-admonition-title-background--warning": "rgba(183, 121, 39, 0.2)",

        "color-admonition-title--danger": "#e15953",  # red-600
        "color-admonition-title-background--danger": "rgba(225, 89, 83, 0.2)",

        "color-admonition-title--attention": "#e15953",  # red-600
        "color-admonition-title-background--attention": "rgba(225, 89, 83, 0.2)",

        "color-admonition-title--error": "#e15953",  # red-600
        "color-admonition-title-background--error": "rgba(225, 89, 83, 0.2)",

        "color-admonition-title--hint": "#6a8b55",  # green-600
        "color-admonition-title-background--hint": "rgba(106, 139, 85, 0.2)",

        "color-admonition-title--tip": "#6a8b55",  # green-600
        "color-admonition-title-background--tip": "rgba(106, 139, 85, 0.2)",

        "color-admonition-title--important": "#4590a9",  # blue-600 (substitute for cyan)
        "color-admonition-title-background--important": "rgba(69, 144, 169, 0.2)",

        "color-admonition-title--note": "#4590a9",  # blue-600
        "color-admonition-title-background--note": "rgba(69, 144, 169, 0.2)",

        "color-admonition-title--seealso": "#8d74cc",  # violet-600
        "color-admonition-title-background--seealso": "rgba(141, 116, 204, 0.2)",

        "color-admonition-title--admonition-todo": "#9d9a97",  # gray-500
        "color-admonition-title-background--admonition-todo": "rgba(157, 154, 151, 0.2)",

        # Topics
        "color-topic-title": "#d9a850",  # yellow-400
        "color-topic-title-background": "rgba(217, 168, 80, 0.2)",
    },
    "dark_css_variables": {


        # Base Colors
        "color-foreground-primary": "#dddad7",
        "color-foreground-secondary": "#9d9a97",
        "color-foreground-muted": "#75726f",
        "color-foreground-border": "#615f5d",

        "color-background-primary": "#3d3b38",
        "color-background-secondary": "#2e2c29",
        "color-background-hover": "#4f4d4a",
        "color-background-hover--transparent": "#4f4d4a00",
        "color-background-border": "#615f5d",
        "color-background-item": "#75726f",

        "color-problematic": "#ff9283",

        # Announcements
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#f1f0ef",

        # Brand
        "color-brand-primary": "#d9a850",
        "color-brand-content": "#f6e299",
        "color-brand-visited": "#c4a1f1",

        # API documentation
        "color-api-keyword": "#9d9a97",
        "color-highlight-on-target": "#5f3d16",

        "color-api-added": "#677851",
        "color-api-added-border": "#9bb989",
        "color-api-changed": "#4590a9",
        "color-api-changed-border": "#79bacd",
        "color-api-deprecated": "#d9a850",
        "color-api-deprecated-border": "#9f6520",
        "color-api-removed": "#ff9283",
        "color-api-removed-border": "#cb4245",

        # Inline code
        "color-inline-code-background": "#2e2c29",

        # Search highlight
        "color-highlighted-background": "#176778",
        "color-highlighted-text": "#dddad7",

        # GUI labels
        "color-guilabel-background": "#176778",
        "color-guilabel-border": "#4590a9",
        "color-guilabel-text": "#dddad7",

        # Tables
        "color-table-header-background": "#2e2c29",
        "color-table-border": "#615f5d",

        # Cards
        "color-card-border": "#2e2c29",
        "color-card-background": "#3d3b38",
        "color-card-marginals-background": "#2e2c29",

        # Header
        "color-header-background": "#3d3b38",
        "color-header-border": "#615f5d",
        "color-header-text": "#dddad7",

        # Sidebar
        "color-sidebar-background": "#2e2c29",
        "color-sidebar-background-border": "#615f5d",

        "color-sidebar-brand-text": "#dddad7",
        "color-sidebar-caption-text": "#9d9a97",
        "color-sidebar-link-text": "#9d9a97",
        "color-sidebar-link-text--top-level": "#d9a850",

        "color-sidebar-item-background": "#2e2c29",
        "color-sidebar-item-background--current": "#2e2c29",
        "color-sidebar-item-background--hover": "#4f4d4a",

        "color-sidebar-item-expander-background": "transparent",
        "color-sidebar-item-expander-background--hover": "#4f4d4a",

        "color-sidebar-search-text": "#dddad7",
        "color-sidebar-search-background": "#2e2c29",
        "color-sidebar-search-background--focus": "#3d3b38",
        "color-sidebar-search-border": "#615f5d",
        "color-sidebar-search-icon": "#9d9a97",

        # TOC
        "color-toc-background": "#3d3b38",
        "color-toc-title-text": "#9d9a97",
        "color-toc-item-text": "#9d9a97",
        "color-toc-item-text--hover": "#dddad7",
        "color-toc-item-text--active": "#d9a850",

        # Links
        "color-link": "#f6e299",
        "color-link-underline": "#615f5d",
        "color-link--hover": "#d9a850",
        "color-link-underline--hover": "#9d9a97",
        "color-link--visited": "#c4a1f1",
        "color-link-underline--visited": "#615f5d",
        "color-link--visited--hover": "#9f86d0",
        "color-link-underline--visited--hover": "#9d9a97",

        # Admonitions
        "color-admonition-background": "#4f4d4a",
        "color-admonition-title": "#9d9a97",
        "color-admonition-title-background": "#2e2c29",

        "color-admonition-title--caution": "#ff9100",
        "color-admonition-title-background--caution": "rgba(255, 145, 0, 0.2)",

        "color-admonition-title--warning": "#ff9100",
        "color-admonition-title-background--warning": "rgba(255, 145, 0, 0.2)",

        "color-admonition-title--danger": "#ff9283",  # softened red for dark
        "color-admonition-title-background--danger": "rgba(255, 146, 131, 0.2)",

        "color-admonition-title--attention": "#ff9283",
        "color-admonition-title-background--attention": "rgba(255, 146, 131, 0.2)",

        "color-admonition-title--error": "#ff9283",
        "color-admonition-title-background--error": "rgba(255, 146, 131, 0.2)",

        "color-admonition-title--hint": "#00c852",
        "color-admonition-title-background--hint": "rgba(0, 200, 82, 0.2)",

        "color-admonition-title--tip": "#00c852",
        "color-admonition-title-background--tip": "rgba(0, 200, 82, 0.2)",

        "color-admonition-title--important": "#00bfa5",
        "color-admonition-title-background--important": "rgba(0, 191, 165, 0.2)",

        "color-admonition-title--note": "#00b0ff",
        "color-admonition-title-background--note": "rgba(0, 176, 255, 0.2)",

        "color-admonition-title--seealso": "#448aff",
        "color-admonition-title-background--seealso": "rgba(68, 138, 255, 0.2)",

        "color-admonition-title--admonition-todo": "#9d9a97",
        "color-admonition-title-background--admonition-todo": "rgba(157, 154, 151, 0.2)",

        # Topics
        "color-topic-title": "#f6e299",
        "color-topic-title-background": "rgba(246, 226, 153, 0.2)",
    },
    "footer_icons": [
        {
            "name": "BertBlocks",
            "url": "https://github.com/coral-nlp/bertblocks",
            "html": """<svg width="100%" height="100%" viewBox="0 0 112 104" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" xmlns:serif="http://www.serif.com/" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linejoin:round;stroke-miterlimit:2;">
                <g transform="matrix(1,0,0,1,-1352.31,-707.902)">
                    <g transform="matrix(1,0,0,1,0,-0.25491)">
                        <g>
                            <g transform="matrix(1.5565,-0.778248,-1.11022e-16,2.81696,-1070.07,-1713.93)">
                                <path d="M1556.3,1305.12L1556.3,1299.43C1556.3,1300.48 1557.99,1302.08 1561.24,1303.87L1581.8,1315.23C1587.47,1318.37 1595.77,1320.91 1600.3,1320.91L1624.98,1320.91C1626.91,1320.91 1627.86,1320.45 1627.86,1319.67L1627.86,1325.35C1627.86,1326.13 1626.91,1326.59 1624.98,1326.59L1600.3,1326.59C1595.77,1326.59 1587.47,1324.05 1581.8,1320.91L1561.24,1309.55C1557.99,1307.76 1556.3,1306.16 1556.3,1305.12Z" style="fill:rgb(217,168,80);"/>
                            </g>
                            <g transform="matrix(1.5565,-0.778248,-1.11022e-16,2.81696,-1070.07,-1697.93)">
                                <path d="M1556.3,1305.12L1556.3,1299.43C1556.3,1300.48 1557.99,1302.08 1561.24,1303.87L1581.8,1315.23C1587.47,1318.37 1595.77,1320.91 1600.3,1320.91L1624.98,1320.91C1626.91,1320.91 1627.86,1320.45 1627.86,1319.67L1627.86,1325.35C1627.86,1326.13 1626.91,1326.59 1624.98,1326.59L1600.3,1326.59C1595.77,1326.59 1587.47,1324.05 1581.8,1320.91L1561.24,1309.55C1557.99,1307.76 1556.3,1306.16 1556.3,1305.12Z" style="fill:rgb(183,121,39);"/>
                            </g>
                            <g transform="matrix(1.5565,-0.778248,-1.11022e-16,2.81696,-1070.07,-1681.93)">
                                <path d="M1556.3,1305.12L1556.3,1299.43C1556.3,1300.48 1557.99,1302.08 1561.24,1303.87L1581.8,1315.23C1587.47,1318.37 1595.77,1320.91 1600.3,1320.91L1624.98,1320.91C1626.91,1320.91 1627.86,1320.45 1627.86,1319.67L1627.86,1325.35C1627.86,1326.13 1626.91,1326.59 1624.98,1326.59L1600.3,1326.59C1595.77,1326.59 1587.47,1324.05 1581.8,1320.91L1561.24,1309.55C1557.99,1307.76 1556.3,1306.16 1556.3,1305.12Z" style="fill:rgb(130,84,32);"/>
                            </g>
                        </g>
                        <g>
                            <g transform="matrix(0.894427,-0.447214,1.11803,0.559017,0,0)">
                                <path d="M-0,1273.66L-0,1302.29C-0,1310.18 -6.412,1316.6 -14.311,1316.6L-57.243,1316.6C-65.142,1316.6 -71.554,1310.18 -71.554,1302.29L-71.554,1273.66C-71.554,1265.77 -65.142,1259.35 -57.243,1259.35L-14.311,1259.35C-6.412,1259.35 -0,1265.77 -0,1273.66Z" style="fill:rgb(246,226,153);"/>
                            </g>
                            <g transform="matrix(1,0,0,1,0,0.25491)">
                                <path d="M1440.32,739.998C1440.36,740.333 1440.35,740.67 1440.29,741.007C1439.88,743.419 1437.58,745.67 1433.4,747.762L1418.72,755.1C1418.47,755.228 1418.17,755.282 1417.85,755.264C1417.52,755.246 1417.2,755.157 1416.88,754.998L1371.19,732.151C1370.87,731.992 1370.69,731.83 1370.65,731.667C1370.65,731.646 1370.65,731.626 1370.65,731.606L1370.65,730.761C1370.65,730.78 1370.65,730.801 1370.65,730.821C1370.68,730.953 1370.8,731.083 1371.02,731.213C1371.07,731.244 1371.12,731.274 1371.19,731.305L1416.88,754.152C1417.2,754.311 1417.52,754.4 1417.85,754.418C1418.17,754.436 1418.47,754.382 1418.72,754.255L1433.4,746.916C1437.58,744.825 1439.88,742.573 1440.29,740.161C1440.3,740.107 1440.31,740.052 1440.32,739.998Z" style="fill:rgb(75,51,19);"/>
                            </g>
                            <g transform="matrix(1,0,0,1,0,0.25491)">
                                <path d="M1405.61,730.937C1405.94,730.166 1405.37,729.409 1403.88,728.666C1401.38,727.416 1398.8,727.454 1396.15,728.782L1391.67,731.019C1391.49,731.11 1391.51,731.212 1391.74,731.326L1392.25,731.578L1396.15,729.628C1398.8,728.3 1401.38,728.261 1403.88,729.512C1404.82,729.981 1405.39,730.456 1405.61,730.937Z" style="fill:rgb(75,51,19);"/>
                            </g>
                            <g transform="matrix(1,0,0,1,0,0.25491)">
                                <path d="M1424.23,739.93C1424.58,739.122 1423.92,738.304 1422.27,737.477C1419.72,736.204 1417.17,736.204 1414.63,737.477L1409.61,739.987C1409.43,740.078 1409.45,740.18 1409.68,740.294L1410.18,740.546L1414.63,738.323C1417.17,737.05 1419.72,737.05 1422.27,738.323C1423.35,738.863 1424,739.398 1424.23,739.93Z" style="fill:rgb(75,51,19);"/>
                            </g>
                            <g transform="matrix(0.894427,-0.447214,1.11803,0.559017,0,0)">
                                <path d="M-51.384,1267.38C-51.359,1267.17 -51.28,1267 -51.149,1266.87C-51.129,1266.85 -51.107,1266.83 -51.085,1266.81C-50.93,1266.69 -50.727,1266.62 -50.478,1266.62L-34.618,1266.62C-29.413,1266.62 -25.499,1267.55 -22.876,1269.4C-20.253,1271.25 -18.941,1274.35 -18.941,1278.7C-18.941,1282.19 -20.2,1284.82 -22.719,1286.58C-22.991,1286.77 -23.276,1286.95 -23.577,1287.12C-23.715,1287.2 -23.759,1287.29 -23.71,1287.37C-23.687,1287.41 -23.643,1287.45 -23.577,1287.49C-21.747,1288.5 -20.405,1289.81 -19.551,1291.39C-18.697,1292.98 -18.27,1294.89 -18.27,1297.12C-18.27,1301.22 -19.669,1304.25 -22.466,1306.23C-22.53,1306.28 -22.596,1306.32 -22.662,1306.37C-25.59,1308.34 -29.392,1309.33 -34.069,1309.33L-50.478,1309.33C-50.763,1309.33 -50.986,1309.24 -51.149,1309.08C-51.312,1308.92 -51.393,1308.69 -51.393,1308.41L-51.393,1267.54C-51.393,1267.49 -51.39,1267.43 -51.384,1267.38ZM-31.739,1282.6C-30.69,1282.07 -30.165,1281.13 -30.165,1279.8C-30.165,1277.56 -31.649,1276.45 -34.618,1276.45L-39.62,1276.45C-39.823,1276.45 -39.925,1276.55 -39.925,1276.75L-39.925,1282.85C-39.925,1283.05 -39.823,1283.16 -39.62,1283.16L-34.618,1283.16C-33.414,1283.16 -32.454,1282.97 -31.739,1282.6ZM-31.385,1298.98C-30.287,1298.41 -29.738,1297.38 -29.738,1295.9C-29.738,1293.63 -31.161,1292.49 -34.008,1292.49L-39.62,1292.49C-39.823,1292.49 -39.925,1292.59 -39.925,1292.79L-39.925,1299.2C-39.925,1299.4 -39.823,1299.5 -39.62,1299.5L-34.069,1299.5C-32.962,1299.5 -32.067,1299.33 -31.385,1298.98Z" style="fill:rgb(159,101,32);"/>
                            </g>
                        </g>
                    </g>
                </g>
            </svg>""",  # truncated for brevity
            "class": "",
        },
    ],
}

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "datasets": ("https://huggingface.co/docs/datasets/main/en/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
}
