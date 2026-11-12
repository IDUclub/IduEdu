# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| iduedu/\_\_init\_\_.py                                |        4 |        0 |        0 |        0 |    100% |           |
| iduedu/\_api.py                                       |       32 |       13 |        2 |        0 |     56% |86-89, 104-113 |
| iduedu/\_version.py                                   |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/config.py                                      |      211 |       11 |       70 |        5 |     94% |14-15, 37-39, 47-48, 133-134, 157, 392, 448-\>exit |
| iduedu/constants/\_\_init\_\_.py                      |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/highway\_enums.py                    |       28 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/network\_enums.py                    |       10 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/transport\_specs.py                  |      122 |        2 |       54 |        2 |     98% |   95, 108 |
| iduedu/graph/\_\_init\_\_.py                          |       15 |        0 |        0 |        0 |    100% |           |
| iduedu/graph/adapters.py                              |      113 |       22 |       34 |        8 |     77% |16, 97, 109-111, 115-116, 125-126, 143-144, 150-157, 210-216 |
| iduedu/graph/adjacency.py                             |       47 |        3 |       18 |        3 |     91% |14, 86, 107 |
| iduedu/graph/aggregations.py                          |      123 |        2 |       28 |        2 |     97% |  325, 330 |
| iduedu/graph/centrality.py                            |      108 |        3 |       38 |        3 |     96% |45, 128, 233 |
| iduedu/graph/components.py                            |       67 |       10 |       16 |        5 |     82% |15, 85-91, 106, 115, 124, 143-144, 176-\>178, 181, 184 |
| iduedu/graph/editors.py                               |      462 |       47 |      222 |       46 |     86% |70, 72, 74, 76, 81, 83, 92, 94, 96, 101, 107, 109, 114, 116, 157, 159, 242-\>257, 252-\>257, 282, 295, 331, 336-\>341, 373, 375, 379, 381, 383, 385, 402, 404, 414, 503, 537, 541-542, 569, 738, 748-750, 753-759, 762-\>765, 768-\>714, 770-\>714, 777-785, 787-\>792, 793-\>806, 869, 888-\>891 |
| iduedu/graph/edits.py                                 |      411 |       17 |      196 |       19 |     94% |67, 264, 283, 285, 320-\>318, 358-\>360, 373, 396-\>395, 399, 430, 434-436, 445, 518-\>522, 527, 541, 556, 560, 562, 571 |
| iduedu/graph/elevation.py                             |      238 |       15 |       68 |       11 |     90% |41-44, 48, 59, 61-\>63, 85, 102, 130, 187, 292-293, 350, 379, 383 |
| iduedu/graph/graph\_inputs.py                         |      125 |       22 |       76 |       22 |     78% |47, 63, 65, 70, 72, 111, 164, 166, 168, 175, 199, 201, 204, 206, 210, 214, 216, 218, 221, 225, 230, 232 |
| iduedu/graph/io.py                                    |      149 |       32 |       52 |       12 |     72% |41-48, 83, 121-122, 175-176, 178-\>181, 212, 218, 223, 225, 232-234, 264-\>263, 270, 276, 282, 285-286, 290-296 |
| iduedu/graph/lines.py                                 |       17 |        0 |        4 |        0 |    100% |           |
| iduedu/graph/nx\_utils.py                             |      175 |       21 |       54 |       14 |     83% |34, 86-87, 92-\>95, 99-\>101, 122-\>127, 125-126, 127-\>121, 130-131, 154-159, 165-166, 198-199, 230-\>239, 281, 290, 295, 297-\>300, 302-\>315, 317 |
| iduedu/graph/point\_to\_point.py                      |       42 |        1 |       14 |        1 |     96% |       157 |
| iduedu/graph/routing.py                               |      406 |       12 |      154 |       16 |     95% |124, 139, 143, 147, 171, 178, 180, 182, 217, 275-\>272, 281, 304-\>306, 616-\>618, 764-\>774, 905, 907 |
| iduedu/graph/routing\_result.py                       |      158 |        6 |       44 |        3 |     96% |17, 35-37, 124, 169 |
| iduedu/graph/segments.py                              |       30 |        1 |        4 |        1 |     94% |        63 |
| iduedu/graph/shortest\_paths.py                       |       64 |        8 |       34 |        8 |     84% |52, 54, 56, 61-62, 71, 75, 77, 102-\>101 |
| iduedu/graph/stats.py                                 |      302 |       10 |       74 |       15 |     93% |32-\>35, 45, 53, 70, 89, 196, 260, 272-\>277, 301-\>313, 309-\>311, 357, 390, 416, 422-\>428, 566 |
| iduedu/graph/transformers.py                          |      156 |       13 |       54 |       18 |     85% |57, 94, 97-\>106, 103, 130, 170, 172, 174, 177, 188, 190, 193-194, 211-\>214, 280-\>291, 312-\>338, 315, 331-\>338, 333-\>336 |
| iduedu/graph/urban\_graph.py                          |      271 |       32 |       38 |        7 |     87% |106-112, 132-\>134, 223, 331-333, 342-344, 364-366, 385, 583-585, 723-725, 742-744, 773, 865, 928, 964-966, 974-976, 986-988, 1041, 1092-1094 |
| iduedu/graph/validation.py                            |       89 |       16 |       60 |       16 |     79% |39, 41, 55, 60, 62, 74, 80, 82, 85, 87, 90, 93, 95, 101, 103, 115 |
| iduedu/graph/within.py                                |       60 |        5 |       26 |        5 |     88% |31, 62, 78, 82, 143 |
| iduedu/graph\_builders/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| iduedu/graph\_builders/bike\_builders.py              |      113 |        5 |       26 |        2 |     95% |181-182, 204-205, 245 |
| iduedu/graph\_builders/drive\_walk\_builders.py       |      198 |        5 |       54 |        6 |     96% |47, 155, 285-\>295, 296, 310, 319 |
| iduedu/graph\_builders/gtfs\_builders.py              |      567 |       63 |      230 |       52 |     85% |62, 64, 69-71, 78, 80-82, 85, 89-90, 92, 101-102, 112-113, 115-\>117, 117-\>119, 128-\>135, 131-\>133, 137-139, 169-\>171, 190, 206-208, 288, 314, 328, 332-336, 342, 368, 391, 400-401, 438-439, 446, 462-\>473, 475, 479-480, 495, 505, 511, 523, 534, 598, 601-\>599, 603, 610, 628, 668, 725-\>736, 755-\>758, 760, 829, 865, 928, 930, 940, 944, 959-961, 970-971 |
| iduedu/graph\_builders/intermodal\_builders.py        |      164 |       10 |       62 |       11 |     91% |30-\>36, 32, 35, 52, 91, 149-\>174, 194-195, 244, 268-\>270, 332, 351-352 |
| iduedu/graph\_builders/public\_transport\_builders.py |      274 |        9 |      104 |       10 |     95% |70, 75-76, 116, 133, 192, 304, 396, 486-\>489, 491, 513-\>522, 517-\>519, 519-\>522 |
| iduedu/gtfs/\_\_init\_\_.py                           |        3 |        0 |        0 |        0 |    100% |           |
| iduedu/gtfs/merge.py                                  |      203 |       16 |      100 |       12 |     90% |48, 121-\>123, 166, 211, 235-245, 250-\>252, 259, 262, 322, 355-359, 431, 438 |
| iduedu/gtfs/reader.py                                 |       56 |        6 |       22 |        4 |     87% |53, 56, 59, 78-79, 113 |
| iduedu/gtfs/service\_days.py                          |       68 |        5 |       32 |        5 |     90% |49, 60, 78, 104, 107 |
| iduedu/gtfs/validation.py                             |       57 |        6 |       30 |        6 |     86% |43, 49, 57, 59, 65, 104 |
| iduedu/osm\_pbf/\_\_init\_\_.py                       |        5 |        0 |        0 |        0 |    100% |           |
| iduedu/osm\_pbf/download.py                           |      142 |       28 |       42 |        8 |     77% |79-82, 119-124, 132-133, 135-136, 151-153, 162-165, 203, 208-209, 212-214, 258, 260-\>262, 270 |
| iduedu/osm\_pbf/extract.py                            |      157 |        3 |       74 |        6 |     96% |175, 179-\>184, 185, 234-\>233, 254, 265-\>264 |
| iduedu/osm\_pbf/filters.py                            |      138 |        9 |       50 |        6 |     92% |82, 103, 109, 114, 121, 133-135, 195 |
| iduedu/osm\_pbf/reader.py                             |      331 |       14 |      108 |       14 |     93% |91-94, 121, 131, 142-\>144, 186-\>200, 191-\>193, 197-\>199, 217-\>219, 237-\>239, 263, 283, 358-362, 369-\>374, 470-\>468, 500, 510-511 |
| iduedu/overpass/\_\_init\_\_.py                       |        0 |        0 |        0 |        0 |    100% |           |
| iduedu/overpass/cache.py                              |       57 |       16 |       10 |        1 |     72% | 59-78, 86 |
| iduedu/overpass/downloaders.py                        |      262 |       68 |       96 |       10 |     74% |37-46, 99-108, 118-122, 137-166, 180-\>182, 182-\>185, 229-238, 275, 389, 396, 495, 533, 558, 563-588 |
| iduedu/overpass/parsers.py                            |      750 |       56 |      344 |       51 |     89% |86, 149, 292-\>331, 311-312, 323, 325-327, 329-\>292, 430, 496-497, 530, 532, 545, 551-\>543, 589-590, 673, 685-686, 695, 705-\>723, 726, 755, 767, 853-\>855, 860-\>862, 878-879, 882, 886, 897, 911, 917-\>951, 943-\>949, 944-\>943, 955, 971, 983, 990, 1025, 1048, 1157-1158, 1159-\>1163, 1165, 1220, 1242-1243, 1273-1274, 1277-1278, 1283-\>1281, 1293-\>1292, 1320-1322, 1325, 1340, 1343-1344, 1349, 1353, 1356-1357 |
| **TOTAL**                                             | **7582** |  **643** | **2818** |  **446** | **89%** |           |


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