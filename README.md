# ХасБанк сорил 2024: RAG

Энэхүү repository-д RAG хөгжүүлэхэд хэрэглэгдэх өгөгдөл болон гарааны код байршина.
Тэмцээний төгсгөлд та бүхэн өөрсдийн хийсэн бүтээлээ доор бичсэн бүтээлд тавигдах шаардлагын дагуу илгээнэ.

## Өгөгдөл

`data` хавтсанд [xacbank.mn](https://www.xacbank.mn/) цахим хуудаснаас авсан ХасБанкны тухай хуудсууд, мэдээ, бүтээгдээхүүн, үйлчилгээний танилцуулга, салбарын мэдээллийг агуулсан `csv` файлуудыг байрлуулсан.

## Гарааны код

RAG хэрэгжүүлсэн гарааны кодыг [`src/rag-starter.ipynb`](src/rag-starter.ipynb) jupyter notebook-д байршуулсан бөгөөд RAG хөгжүүлэх суурь болгон ашиглаж болно.

Гарааны кодыг ажиллуулахад шаардлагатай сангуудыг [`requirements.txt`](requirements.txt) файлд жагсаасан.

Сервер дээр гарааны кодыг ажиллуулахдаа repository-г clone хийж аваад `cd` командаар татаж авсан хавтас руугаа ороод, доорх байдлаар python virtual environment үүсгэн ажиллуулж болно:

```sh
# virtual environment үүсгэх
python3 -m venv .venv

# virtual env идэвхжүүлэх
source .venv/bin/activate

# Шаардлагатай сангуудыг суулгах
pip install -r requirements.txt

# Jupyter сервер асаах
jupyter-lab --ip 0.0.0.0
```

Гарч ирсэн `http://<server-ip>:8888/lab?token=...` гэсэн холбоосын дагуу browser-оосоо хандахад Jupyter notebook ажиллуулах орчин гарч ирнэ.

Жич: Гарааны код ажиллуулах, өөрсдийн хэрэгжүүлэлтээ хийх зэрэг ажлыг сервер дээр гүйцэтгэхдээ зөвхөн jupyter lab-р хязгаарлахгүй, өөрт тохирсон байдлаар ажиллаж болно. Жишээ нь VS Code SSH remote development.

## Хакатоны бүтээлд тавигдах шаардлага

Хакатоны багаар ажиллах хугацааны төгсгөлд буюу танилцуулга илтгэл тавихаас өмнө бүх баг өөрсдийн хөгжүүлсэн системийн код болон холбоотой файлуудыг энэхүү repository-д оруулж, commit хийнэ.

**Бүтээлийн шаардлага:**

- RAG системийг ажиллуулахад шаардлагатай бүх **файл, эх кодыг оруулсан байх**
- Системийг ажиллуулахад шаардлагатай хэрэглэх заавар болон тайлбар оруулсан байх
- Модель сургасан бол Hugging Face-д оруулж, холбоосыг GitHub repository дотроо зааж өгөх
- Pitch илтгэлийнхээ presentation-ийг **PDF хэлбэрээр [presentation/](presentation) хавтсанд** байрлуулах
- Pitch илтгэлийг PDF хэлбэрээр үзэхэд асуудалгүй байх (animation-тай хэсгүүд давхардах, уншигдахгүй болох зэрэг)