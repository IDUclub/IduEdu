# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/IDUclub/IduEdu/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ------: | --------: |
| iduedu/\_\_init\_\_.py                                |        4 |        0 |        0 |        0 |    100% |           |
| iduedu/\_api.py                                       |       29 |       13 |        2 |        0 |     52% |52-55, 70-79 |
| iduedu/\_numba/\_\_init\_\_.py                        |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/\_numba/components.py                          |       87 |        3 |       34 |        2 |     96% | 10, 16-17 |
| iduedu/\_numba/csr.py                                 |       52 |        1 |        4 |        0 |     98% |        47 |
| iduedu/\_numba/shortest\_paths.py                     |      247 |       15 |       98 |       11 |     92% |32, 47-102, 143-\>142, 165-\>164, 172, 189-\>181, 214, 264-\>262, 278-\>277, 295, 303-\>302 |
| iduedu/\_version.py                                   |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/config.py                                      |      127 |        4 |       42 |        3 |     96% |74, 224, 276-\>exit, 307-308 |
| iduedu/constants/\_\_init\_\_.py                      |        1 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/highway\_enums.py                    |       28 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/network\_enums.py                    |        9 |        0 |        0 |        0 |    100% |           |
| iduedu/constants/transport\_specs.py                  |      104 |        2 |       40 |        2 |     97% |    78, 91 |
| iduedu/graph/\_\_init\_\_.py                          |        9 |        0 |        0 |        0 |    100% |           |
| iduedu/graph/adapters.py                              |      113 |       22 |       34 |        8 |     77% |16, 97, 109-111, 115-116, 125-126, 143-144, 150-157, 210-216 |
| iduedu/graph/adjacency.py                             |       70 |        6 |       26 |        5 |     89% |37, 43, 106, 113, 126-127 |
| iduedu/graph/components.py                            |       65 |       10 |       16 |        5 |     81% |16, 82-88, 105, 114, 123, 142-143, 175-\>177, 180, 183 |
| iduedu/graph/editors.py                               |      377 |       44 |      180 |       42 |     84% |61, 63, 65, 67, 72, 74, 83, 85, 87, 92, 98, 100, 120-\>131, 125-\>131, 128, 156, 169, 202, 208-\>213, 245, 247, 251, 253, 255, 257, 274, 276, 286, 375, 409, 413-414, 441, 609, 619-621, 624-630, 633-\>636, 639-\>585, 641-\>585, 648-656, 658-\>663, 664-\>677, 721 |
| iduedu/graph/graph\_inputs.py                         |       74 |       24 |       52 |       23 |     63% |35, 37, 39, 41, 45, 47, 49, 53, 55, 66, 90, 92, 95, 97, 101, 105, 107, 109, 112, 116, 121, 123, 128-129 |
| iduedu/graph/io.py                                    |      149 |       32 |       52 |       12 |     72% |41-48, 83, 121-122, 175-176, 178-\>181, 212, 218, 223, 225, 232-234, 264-\>263, 270, 276, 282, 285-286, 290-296 |
| iduedu/graph/nx\_utils.py                             |      173 |       21 |       54 |       14 |     83% |34, 86-87, 92-\>95, 99-\>101, 122-\>127, 125-126, 127-\>121, 130-131, 154-159, 165-166, 198-199, 230-\>239, 281, 290, 295, 297-\>300, 302-\>315, 317 |
| iduedu/graph/shortest\_paths.py                       |      251 |       22 |       88 |       23 |     87% |62, 64, 94, 96, 98, 100, 102, 104, 114, 133, 316-\>319, 484, 558-\>561, 567-568, 576-\>579, 731, 733, 735, 740-741, 750, 754, 756, 766 |
| iduedu/graph/transformers.py                          |       71 |       12 |       30 |       13 |     75% |51, 88, 91-\>100, 97, 124, 164, 166, 168, 171, 182, 184, 187-188, 205-\>208 |
| iduedu/graph/urban\_graph.py                          |      207 |       32 |       36 |        8 |     83% |92-98, 118-\>120, 138, 213, 321-323, 332-334, 354-356, 375, 445-447, 505-507, 536-538, 569-571, 624-626, 643-645, 674, 727, 765, 828, 909 |
| iduedu/graph/validation.py                            |       89 |       19 |       60 |       21 |     73% |11, 29-30, 39, 41, 55, 58-\>exit, 60, 62, 74, 80, 82, 85, 87, 90, 93, 95, 99-\>exit, 101, 103, 111-\>117, 115 |
| iduedu/graph\_builders/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| iduedu/graph\_builders/drive\_walk\_builders.py       |      188 |        4 |       52 |        4 |     97% |47, 277, 291, 300 |
| iduedu/graph\_builders/gtfs\_builders.py              |      488 |       55 |      198 |       48 |     84% |61, 63, 68-70, 77, 79-81, 84, 88-89, 91, 100-101, 111-112, 114-\>116, 116-\>118, 127-\>134, 130-\>132, 136-138, 168-\>170, 189, 205-207, 281, 295, 299-303, 309, 340-341, 343, 348, 364-\>375, 377, 381-382, 397, 407, 413, 425, 436, 500, 503-\>501, 505, 512, 530, 570, 625-\>627, 627-\>638, 657-\>660, 662, 731, 762, 767, 814, 816, 829-830 |
| iduedu/graph\_builders/intermodal\_builders.py        |      148 |        8 |       58 |       10 |     91% |29-\>35, 31, 34, 51, 90, 148-\>173, 223, 247-\>249, 311, 330-331 |
| iduedu/graph\_builders/public\_transport\_builders.py |      246 |       13 |       96 |       12 |     93% |39, 44-45, 85, 102, 150, 262, 302, 353, 440, 444, 466-\>473, 468-\>470, 470-\>473, 485-486 |
| iduedu/gtfs/\_\_init\_\_.py                           |        3 |        0 |        0 |        0 |    100% |           |
| iduedu/gtfs/merge.py                                  |      134 |        9 |       70 |        7 |     91% |48, 121-\>123, 160, 209-213, 277, 284, 290-\>292 |
| iduedu/gtfs/reader.py                                 |       56 |        6 |       22 |        4 |     87% |53, 56, 59, 78-79, 113 |
| iduedu/gtfs/validation.py                             |       57 |        8 |       30 |       11 |     78% |43, 49, 57, 59, 65, 74-\>76, 77, 91-\>93, 94, 100-\>exit, 104 |
| iduedu/overpass/\_\_init\_\_.py                       |        0 |        0 |        0 |        0 |    100% |           |
| iduedu/overpass/cache.py                              |       57 |        4 |       10 |        2 |     91% |63, 68-69, 86 |
| iduedu/overpass/downloaders.py                        |      219 |       12 |       76 |       11 |     92% |74-78, 136-\>138, 138-\>141, 201-204, 234, 451, 455-\>457, 458, 459-\>462, 490 |
| iduedu/overpass/parsers.py                            |      736 |       57 |      340 |       52 |     89% |85, 279-\>318, 298-299, 310, 312-314, 316-\>279, 417, 456-\>453, 483-484, 517, 519, 532, 538-\>530, 576-577, 662, 674-675, 684, 694-\>712, 715, 741, 753, 760, 820-\>822, 827-\>829, 845-846, 849, 853, 864, 878, 884-\>918, 910-\>916, 911-\>910, 922, 938, 950, 957, 992, 1015, 1108, 1113-1114, 1121, 1176, 1198-1199, 1229-1230, 1233-1234, 1239-\>1237, 1249-\>1248, 1276-1278, 1281, 1296, 1299-1300, 1305, 1309, 1312-1313 |
| **TOTAL**                                             | **4670** |  **458** | **1800** |  **353** | **87%** |           |


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