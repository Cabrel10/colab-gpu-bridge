# Colab GPU Bridge — notebook d'origine modernisé avec RavenX 27B

Le notebook joint d'origine est versionné sous [`notebooks/genspark_code.ipynb`](notebooks/genspark_code.ipynb). Ses cellules et identifiants Colab ont été conservés, tandis que les deux sélecteurs concurrents et leurs entrées Mistral/Qwen3.5 ont été remplacés par un pipeline unique pour **RavenX Chaos Agent Qwen3.8 27B Q4_K_M**. Une seule instance `llama_cpp.Llama` sert l'inférence directe et l'API OpenAI compatible.

[Ouvrir le notebook d'origine modifié dans Google Colab](https://colab.research.google.com/github/Cabrel10/colab-gpu-bridge/blob/genspark_ai_developer/notebooks/genspark_code.ipynb)

`notebooks/RavenX_Chaos_Agent_Colab.ipynb` reste un prototype historique ; ce n'est plus le livrable canonique.

## Lancer Colab

1. Ouvrir le lien Colab ci-dessus.
2. Choisir **Runtime → Change runtime type → GPU**.
3. Dans les secrets Colab, créer uniquement :
   - `hex` : chaîne aléatoire d'au moins 16 caractères, utilisée comme clé Bearer et secret HMAC ;
   - `vps` : `IP:PORT` ou URL HTTPS du récepteur `/heartbeat`.
4. Autoriser l'accès du notebook à ces deux secrets.
5. Choisir **Runtime → Run all**.
6. Attendre le JSON final contenant `"status": "READY"` et `openai_base`.

Le notebook diagnostique T4/L4/A100/H100, installe `llama-cpp-python` avec CUDA, appelle explicitement `Llama.from_pretrained`, valide les 16 547 399 968 octets du GGUF, adapte contexte et offload, exécute une inférence directe, teste `/health`, `/v1/models`, le chat et le streaming SSE, ouvre Cloudflare, puis envoie un heartbeat HMAC toutes les 60 secondes.

Le tunnel `trycloudflare.com` est temporaire : relancer Colab produit généralement une nouvelle URL. Le heartbeat permet au VPS de recevoir cette URL.

## Cache Google Drive et cold start parallèle

Le notebook monte `MyDrive/colab_llm_cache` et y conserve deux caches persistants :

- le GGUF, accepté uniquement si sa taille exacte est `16 547 399 968` octets ;
- le cache `pip`, réutilisé par les sessions Colab suivantes.

Pendant le démarrage, un `ThreadPoolExecutor(max_workers=2)` lance simultanément :

1. l'installation de l'environnement CUDA et des dépendances Python ;
2. la copie du GGUF depuis Drive, ou son téléchargement avec reprise (`curl -C -`) lors de la première session.

Le fichier partiel n'est déplacé vers le chemin actif puis copié dans Drive qu'après validation de sa taille. La première session reste contrainte par l'installation et le téléchargement ; les suivantes suivent le fast path Drive et évitent de télécharger à nouveau les 15 Go. Les durées restent indicatives et dépendent du débit Drive, du GPU attribué et de la disponibilité des wheels CUDA.

## Modèle et dataset examiné

- Dépôt d'inférence : `deadbydawn101/RavenXAiLabs-Chaos-Agent-Qwen3.8-27B-Frontier-Intelligence-Injected-OBLITERATED-GGUF`
- Fichier : `RavenX-Chaos-Agent-Q4_K_M.gguf`
- Alias API : `ravenx-chaos-agent-qwen3.8-27b`
- Dataset examiné : `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`

Le dépôt r0b0tlab contient environ 57 937 traces SFT Parquet issues de plusieurs enseignants. Ce n'est pas un modèle GGUF et il n'est pas chargeable directement pour l'inférence. Il pourra servir ultérieurement à une opération distincte de fine-tuning ou de distillation.

## Client retenu : OpenCode TUI/CLI

OpenCode est retenu car il sait utiliser une API OpenAI compatible et fournit les outils de lecture, édition, recherche, shell et MCP. Les outils s'exécutent sur la machine qui lance OpenCode (poste local ou VPS), tandis que le LLM tourne dans Colab.

### Installation

```bash
curl -fsSL https://opencode.ai/install | bash
# alternative : npm install -g opencode-ai
```

### Configuration du projet

Le fichier [`opencode.json`](opencode.json) déclare le provider RavenX et le serveur MCP distant Context7. Dans le terminal où OpenCode sera lancé :

```bash
export RAVENX_BASE_URL='https://URL-AFFICHEE-PAR-COLAB.trycloudflare.com/v1'
export RAVENX_API_KEY='valeur-exacte-du-secret-hex'
cd /chemin/du/projet
cp /chemin/vers/colab-gpu-bridge/opencode.json ./opencode.json
```

Ne jamais committer `RAVENX_API_KEY`. Le fichier JSON ne contient que les substitutions `{env:RAVENX_BASE_URL}` et `{env:RAVENX_API_KEY}`.

### TUI interactive

```bash
opencode . --model ravenx/ravenx-chaos-agent-qwen3.8-27b
```

Dans la TUI, `/models` doit afficher RavenX. OpenCode demandera confirmation avant les opérations sensibles selon sa politique locale.

### CLI non interactive

```bash
opencode models ravenx
opencode run --model ravenx/ravenx-chaos-agent-qwen3.8-27b \
  "Lis le projet, exécute les tests et résume les échecs sans modifier les fichiers."
```

### Depuis un VPS par SSH

```bash
ssh user@vps
cd /srv/mon-projet
export RAVENX_BASE_URL='https://URL-AFFICHEE-PAR-COLAB.trycloudflare.com/v1'
read -rsp 'RavenX API key: ' RAVENX_API_KEY && export RAVENX_API_KEY && echo
opencode . --model ravenx/ravenx-chaos-agent-qwen3.8-27b
```

Le secret saisi avec `read -s` n'est ni affiché ni stocké dans l'historique shell.

### IDE

OpenCode reste le moteur agent principal. Depuis un terminal intégré VS Code, Zed ou JetBrains, exécuter la même commande `opencode .`. Pour un éditeur compatible ACP, lancer `opencode acp`; le modèle et les outils restent ceux du fichier `opencode.json`.

## Vérifications API

```bash
curl -fsS "$RAVENX_BASE_URL/models" \
  -H "Authorization: Bearer $RAVENX_API_KEY"

curl -fsS "$RAVENX_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $RAVENX_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"ravenx-chaos-agent-qwen3.8-27b","messages":[{"role":"user","content":"Réponds uniquement API_OK"}],"temperature":0,"max_tokens":32}'
```

## MCP et preuve des outils

`opencode.json` active Context7 sans secret. Vérifier sa connexion :

```bash
opencode mcp list
opencode run --model ravenx/ravenx-chaos-agent-qwen3.8-27b \
  "Utilise context7 pour vérifier la documentation FastAPI, puis crée /tmp/ravenx-proof.txt avec une synthèse et affiche le fichier."
test -s /tmp/ravenx-proof.txt && cat /tmp/ravenx-proof.txt
```

Cette preuve couvre un appel MCP, une écriture de fichier et une commande shell. Pour un MCP privé, ajouter un bloc `type: "remote"` avec une URL et un header utilisant `{env:NOM_DU_SECRET}`, jamais une valeur secrète en clair.

## Limites opérationnelles

- Le runtime Colab doit rester actif ; la dernière cellule maintient le service.
- Le contexte annoncé à OpenCode est 4096 tokens, compatible avec le profil T4 minimal. Le notebook utilise 8192 sur L4 et 16384 sur A100/H100.
- Une seule génération est exécutée à la fois afin de protéger l'instance et la VRAM.
- Les outils OpenCode/MCP sont exécutés côté client et restent soumis aux permissions de cette machine.
- Le modèle est publié comme « obliterated » : utiliser des permissions minimales, relire les commandes proposées et ne jamais exposer le tunnel sans clé Bearer.

## Tests statiques

```bash
python -m pytest -q
python -m json.tool notebooks/genspark_code.ipynb >/dev/null
python -m json.tool opencode.json >/dev/null
```

Les tests locaux valident la structure et la syntaxe du notebook canonique issu de la pièce jointe, la conservation de ses identifiants, le modèle unique, l'absence de Mistral/Qwen3.5 et de double sélecteur, les deux secrets, l'authentification, les routes API, le streaming et la configuration OpenCode. L'inférence 27B et le tunnel nécessitent un vrai runtime Colab GPU et ne peuvent pas être reproduits dans le test statique local.
