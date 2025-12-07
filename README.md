# Reinforcement Learning-Based Joint Beamforming Design for RIS-ISAC Vehicular Network with Delayed Channels
25_2_RL_project: 지연된 채널을 가지는 RIS-ISAC 차량 네트워크에서 강화학습 기반 공동 빔포밍 설계

**파일 설명**
\data
    \data\train : MATLAB으로 생성한 채널 데이터셋(.mat 파일) 40개 (모델 훈련용)
    \data\test  : MATLAB으로 생성한 채널 데이터셋(.mat 파일)  1개 (모델 일반화 성능 테스트용)

A2C.py : A2C 에이전트 코드


DDPG.py : DDPG 에이전트 코드


DQN.py : DQN 에이전트 코드
main_RL.py : RL main 환경 코드 (off-policy 모델용)
main_RL_onpolicy.py : : RL main 코드 (on-policy 모델용)
PPO.py : PPO 에이전트 코드
processing.py : 데이터셋 전처리 및 계산용 코드
SAC.py : SAC 에이전트 코드
TD3.py :TD3 에이전트 코드


**필요 라이브러리**
numpy
torch
time <-- 훈련 소요시간 체크용 (불필요 시 제외 가능)


**강화학습 모델 훈련 코드 실행**
1. main_RL.py에서 학습시킬 에이전트를 선택(= 주석 처리 해제) (9-12 line)
2. 파라미터(episode 수, batch size, learning rate, discount factor 등)조정 (23-42 line)
3. train/test data를 불러올 경로를 각각 \data\train, \data\test로 설정 (46-51 line)
4. 코드 실행
예시) SAC 학습 하고 싶다 -> main_RL.py 코드 상단에서 `from SAC import SACAgent as Agent` 활성화, 나머지 에이전트 주석처리
cf) A2C, PPO 학습하고 싶다 -> main_RL_onpolicy.py에서 에이전트를 선택하고, 이후 동일


**코드 실행 시 저장 파일**
\results 폴더에 {agent_name}_result.pt 파일과 {agent_name}_agent.pt 파일 각각 저장 (파일명 임의 수정 가능, 300-320 line)
result.pt : 'reward_ep', 'snrt_ep', 'snrc_ep', 'viol_ep', 'snrt_test', 'snrc_test', 'viol_test', 'total_time_sec' 각각 포함.
  순서대로 에피소드 당 평균 리워드, 에피소드 당 평균 센싱 SNR(dB), 에피소드 당 평균 통신 SNR(dB), 에피소드 당 제약 위반율, 테스트 결과에서 평균 센싱 SNR(dB), 테스트 결과에서 평균 통신 SNR(dB), 테스트 결과에서 위반율, 총 걸린시간
agent.pt : 학습된 에이전트 자체를 저장
  

**학습한 모델 예시**
에이전트를 저장한 파일의 용량이 너무 커서(약 3.4 GB) 아래의 구글 드라이브 링크에 업로드 하였습니다.

https://drive.google.com/drive/folders/1GxMiBQ5mtn1r8Vta38dhyYbxnFbtJnHV?usp=sharing

