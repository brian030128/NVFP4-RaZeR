#!/bin/bash
# Part C is PAUSED before Phi-4 by user decision (2026-09-25): this stub replaces queue_model.sh so that a chain
# reaching it launches nothing. The real queue is queue_model.sh.paused; restore it only on the user's decision.
echo "$(date -u +%FT%TZ) $1 NOT STARTED: Part C paused before Phi-4 by user decision" >> /home/dev/n16k64_campaign/multimodel/commands.log
exit 0
