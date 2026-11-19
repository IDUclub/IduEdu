# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| iduedu/\_\_init\_\_.py                                |        4 |        0 |        0 |        0 |    100% |           |
| iduedu/\_api.py                                       |       33 |       13 |        2 |        0 |     57% |91-94, 109-118 |
| iduedu/\_version.py                                   |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/config.py                                      |      214 |       12 |       70 |        5 |     94% |14-15, 37-39, 47-48, 137-138, 161, 396, 432, 456-\>exit |
| iduedu/constants/\_\_init\_\_.py                      |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/highway\_enums.py                    |       28 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/network\_enums.py                    |       11 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/transport\_specs.py                  |      122 |        2 |       54 |        2 |     98% |   95, 108 |
| iduedu/graph/\_\_init\_\_.py                          |       17 |        0 |        0 |        0 |    100% |           |
| iduedu/graph/adapters.py                              |      121 |       14 |       40 |        6 |     86% |16, 97, 109-111, 115-116, 125-126, 151-158 |
| iduedu/graph/adjacency.py                             |       47 |        3 |       18 |        3 |     91% |14, 86, 107 |
| iduedu/graph/aggregations.py                          |      123 |        2 |       28 |        2 |     97% |  325, 330 |
| iduedu/graph/centrality.py                            |      108 |        3 |       38 |        3 |     96% |45, 128, 233 |
| iduedu/graph/components.py                            |       67 |       10 |       16 |        5 |     82% |15, 85-91, 106, 115, 124, 143-144, 176-\>178, 181, 184 |
| iduedu/graph/editors.py                               |      463 |       45 |      222 |       44 |     86% |71, 73, 75, 77, 82, 84, 93, 95, 97, 102, 108, 110, 115, 117, 158, 160, 243-\>258, 253-\>258, 284, 297, 334, 339-\>344, 377, 379, 383, 385, 387, 389, 408, 508, 542, 546-547, 574, 743, 753-755, 758-764, 767-\>770, 773-\>719, 775-\>719, 782-790, 792-\>797, 798-\>811, 874, 893-\>896 |
| iduedu/graph/edits.py                                 |      411 |       17 |      196 |       19 |     94% |67, 264, 283, 285, 320-\>318, 358-\>360, 373, 396-\>395, 399, 430, 434-436, 445, 518-\>522, 527, 541, 556, 560, 562, 571 |
| iduedu/graph/elevation.py                             |      238 |       15 |       68 |       11 |     90% |41-44, 48, 59, 61-\>63, 85, 102, 130, 187, 292-293, 350, 379, 383 |
| iduedu/graph/graph\_inputs.py                         |      125 |       22 |       76 |       22 |     78% |47, 63, 65, 70, 72, 111, 164, 166, 168, 175, 199, 201, 204, 206, 210, 214, 216, 218, 221, 225, 230, 232 |
| iduedu/graph/io.py                                    |      176 |       22 |       64 |       11 |     84% |48-49, 85, 134-135, 196-197, 199-\>202, 234, 240, 245, 247, 256, 297-\>296, 304-\>303, 322, 325-326, 330-336 |
| iduedu/graph/nx\_utils.py                             |      175 |       21 |       54 |       14 |     83% |34, 86-87, 92-\>95, 99-\>101, 122-\>127, 125-126, 127-\>121, 130-131, 154-159, 165-166, 198-199, 230-\>239, 281, 290, 295, 297-\>300, 302-\>315, 317 |
| iduedu/graph/point\_to\_point.py                      |       42 |        1 |       14 |        1 |     96% |       157 |
| iduedu/graph/routes.py                                |      134 |       10 |       56 |       11 |     89% |53, 57, 87-88, 93, 100, 104-\>106, 134-\>133, 146-147, 203-204, 232-\>236, 237-\>241 |
| iduedu/graph/routing.py                               |      406 |       12 |      154 |       16 |     95% |124, 139, 143, 147, 171, 178, 180, 182, 217, 275-\>272, 281, 304-\>306, 616-\>618, 764-\>774, 905, 907 |
| iduedu/graph/routing\_result.py                       |      158 |        6 |       44 |        3 |     96% |17, 35-37, 124, 169 |
| iduedu/graph/segments.py                              |       30 |        1 |        4 |        1 |     94% |        63 |
| iduedu/graph/shortest\_paths.py                       |       64 |        8 |       34 |        8 |     84% |52, 54, 56, 61-62, 71, 75, 77, 102-\>101 |
| iduedu/graph/simplify.py                              |      124 |        3 |       32 |        3 |     96% |25, 48, 110 |
| iduedu/graph/stats.py                                 |      302 |       10 |       74 |       15 |     93% |32-\>35, 45, 53, 70, 89, 196, 260, 272-\>277, 301-\>313, 309-\>311, 357, 390, 416, 422-\>428, 566 |
| iduedu/graph/transformers.py                          |      156 |       13 |       54 |       18 |     85% |57, 94, 97-\>106, 103, 131, 172, 174, 176, 179, 191, 193, 196-197, 214-\>217, 284-\>295, 316-\>342, 319, 335-\>342, 337-\>340 |
| iduedu/graph/urban\_graph.py                          |      284 |       29 |       38 |        8 |     89% |121, 144-\>146, 237, 346-348, 357-359, 393-395, 400-402, 421, 619-621, 759-761, 778-780, 809, 901, 964, 1000-1002, 1010-1012, 1022-1024, 1077, 1128-1130 |
| iduedu/graph/validation.py                            |       89 |       16 |       60 |       16 |     79% |39, 41, 55, 60, 62, 74, 80, 82, 85, 87, 90, 93, 95, 101, 103, 115 |
| iduedu/graph/within.py                                |       60 |        5 |       26 |        5 |     88% |31, 62, 78, 82, 143 |
| iduedu/graph\_builders/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| iduedu/graph\_builders/bike\_builders.py              |      118 |        5 |       26 |        2 |     95% |201-202, 229-230, 265 |
| iduedu/graph\_builders/drive\_walk\_builders.py       |      214 |        5 |       60 |        6 |     96% |47, 112, 189, 335-\>345, 356, 365 |
| iduedu/graph\_builders/gtfs\_builders.py              |      620 |       65 |      248 |       56 |     85% |64, 66, 71-73, 80, 82-84, 87, 91-92, 94, 103-104, 114-115, 117-\>119, 119-\>121, 130-\>137, 133-\>135, 139-141, 171-\>173, 192, 208-210, 290, 316, 330, 334-338, 344, 370, 393, 402-403, 440-441, 448, 464-\>475, 477, 481-482, 497, 507, 513, 525, 536, 596, 612-\>615, 654, 702, 705-\>703, 707, 714, 732, 773, 832-\>843, 862-\>865, 867, 937-\>939, 941, 979, 1058, 1060, 1070, 1074, 1089-1091, 1100-1101 |
| iduedu/graph\_builders/intermodal\_builders.py        |      164 |       10 |       62 |       11 |     91% |27-\>33, 29, 32, 50, 89, 147-\>172, 192-193, 242, 267-\>269, 331, 350-351 |
| iduedu/graph\_builders/public\_transport\_builders.py |      315 |        9 |      118 |       11 |     95% |76, 88, 93-94, 126, 135-\>134, 156, 201, 276, 403, 594-\>603, 598-\>600, 600-\>603 |
| iduedu/graph\_builders/street\_builders.py            |      149 |       10 |       42 |        8 |     90% |93-\>95, 107-\>109, 179, 196-197, 199-203, 209-\>211, 231, 303 |
| iduedu/gtfs/\_\_init\_\_.py                           |        3 |        0 |        0 |        0 |    100% |           |
| iduedu/gtfs/merge.py                                  |      203 |       16 |      100 |       12 |     90% |48, 121-\>123, 166, 211, 235-245, 250-\>252, 259, 262, 322, 355-359, 431, 438 |
| iduedu/gtfs/reader.py                                 |       56 |        6 |       22 |        4 |     87% |53, 56, 59, 78-79, 113 |
| iduedu/gtfs/service\_days.py                          |       68 |        5 |       32 |        5 |     90% |49, 60, 78, 104, 107 |
| iduedu/gtfs/validation.py                             |       57 |        6 |       30 |        6 |     86% |43, 49, 57, 59, 65, 104 |
| iduedu/osm\_pbf/\_\_init\_\_.py                       |        5 |        0 |        0 |        0 |    100% |           |
| iduedu/osm\_pbf/download.py                           |      148 |       31 |       46 |       10 |     76% |80-83, 120-125, 133-134, 136-137, 152-154, 163-166, 204, 209-210, 213-215, 263, 265-\>272, 268, 270-271, 280 |
| iduedu/osm\_pbf/extract.py                            |      165 |        4 |       84 |        7 |     96% |176, 181, 185-\>190, 191, 248-\>247, 268, 279-\>278 |
| iduedu/osm\_pbf/filters.py                            |      138 |        9 |       50 |        6 |     92% |82, 103, 109, 114, 121, 133-135, 195 |
| iduedu/osm\_pbf/reader.py                             |      341 |       14 |      110 |       14 |     93% |90-93, 122, 131, 142-\>144, 188-\>202, 193-\>195, 199-\>201, 219-\>221, 244-\>246, 270, 290, 366-370, 377-\>382, 497-\>495, 528, 538-539 |
| iduedu/overpass/\_\_init\_\_.py                       |        0 |        0 |        0 |        0 |    100% |           |
| iduedu/overpass/cache.py                              |       57 |       16 |       10 |        1 |     72% | 59-78, 86 |
| iduedu/overpass/downloaders.py                        |      269 |       68 |       98 |       10 |     74% |37-46, 99-108, 118-122, 137-166, 180-\>182, 182-\>185, 229-238, 275, 389, 396, 499, 544, 569, 574-599 |
| iduedu/overpass/parsers.py                            |      509 |       31 |      206 |       24 |     92% |78, 202, 210, 242, 257-\>251, 328-330, 357, 435-436, 460-\>465, 466-\>471, 544, 567, 703-704, 705-\>709, 711, 752, 774-775, 805-806, 815-\>813, 825-\>824, 852-854, 857, 872, 875-876, 881, 885, 888-889 |
| **TOTAL**                                             | **7933** |  **625** | **2880** |  **445** | **89%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/IDUclub/IduEdu/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/IDUclub/IduEdu/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FIDUclub%2FIduEdu%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.