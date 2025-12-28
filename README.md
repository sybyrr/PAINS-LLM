# PAINS-LLM
PAINS Project , KBO specialized LLM

<파이프라인 요약>

query가 들어오면, few example로 학습시킨 LLM 모델로 분류를 진행한다. (general question, season analysis, match analysis)

만약 general question이라면, prompting되어있는 기본 LLM 모델로 답변.

그 외의 경우, RAG 절차를 수행한다. 다음은 RAG의 과정이다.

먼저 Query를 형식에 맞추어 refine한 후 벡터 공간에 매핑한다. 이후 우리의 dataset을 embedding한 벡터와의 유사도를 측정해 ranking하는데, 이때 dataset은 그 자체가 임베딩되는 것이 아닌, 그 데이터셋을 대표하는 영어 텍스트가 임베딩된다. 이렇게 가장 관련이 높아보이는 데이터셋을 선정한다.

LLM에게 데이터셋과 query를 태우기 전에, post-processing 과정을 거쳐 중복되거나 불필요한 헤더 등을 없앤다.

LLM은 최종적으로 refine된 데이터셋을 이용해 query에 대한 답변을 생성한다. 이때 LLM은 답변하기 전, RAG를 통해 제공된 데이터셋이 정말 query와 부합하는지 확인하는 절차를 거친다.

만약 필요하다면 agent를 호출해 대시보드를 만들게 되는데, 이는 우리가 미리 작성해놓은 일련의 알고리즘을 따르게 된다.