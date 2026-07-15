# VESSL 재시작 복구 (한 줄)

VESSL 워크스페이스를 재시작하면 `/root`(torch·pip·스크립트·vfla·dscpkg·체크포인트)가 전부 초기화됨.
영속 마운트 `/root/smaller`(dora-storage)만 생존. 복구:

```bash
bash /root/smaller/sh_rebuild/sh_setup.sh
```

수행 내용: pip 설치(torch/fla deps/transformers==5.12.1/lightning) → vfla·dscpkg 압축해제 →
스크립트 복사 → gdn2-1.3B 체크포인트(있으면 복사, 없으면 재다운로드) → import 검증.

이후 실행 시 항상: `export TRITON_CACHE_DIR=/root/triton_cache HF_HUB_DISABLE_XET=1`

**공개키 재등록**(재시작 시 authorized_keys도 초기화될 수 있음):
VESSL 콘솔에서 `~/.ssh/authorized_keys`에 claude 공개키 추가.

스테이징된 아티팩트: `/root/smaller/sh_rebuild/` (vfla.tgz, dscpkg.tgz, *.py, sh_setup.sh)
