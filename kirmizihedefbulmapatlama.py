import threading
import time
import cv2
import numpy as np
import winsound


def ses_cal(frekans, sure_ms):
  # FPS düşüşünü engellemek için daemon thread 
  threading.Thread(
      target=winsound.Beep, args=(frekans, sure_ms), daemon=True
  ).start()


GENISLIK = 640
YUKSEKLIK = 480
ESIK = 20

ekran_merkez_x = GENISLIK // 2
ekran_merkez_y = YUKSEKLIK // 2
ekran_merkezi = (ekran_merkez_x, ekran_merkez_y)

# Kırmızı Renk Sınırları
alt_kirmizi1 = np.array([0, 120, 70])
ust_kirmizi1 = np.array([10, 255, 255])
alt_kirmizi2 = np.array([170, 120, 70])
ust_kirmizi2 = np.array([180, 255, 255])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, GENISLIK)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, YUKSEKLIK)

kilit_baslangic = None
HEDEF_SURE = 2.0
son_bip_zamani = 0
atis_yapildi = False

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  mask1 = cv2.inRange(hsv, alt_kirmizi1, ust_kirmizi1)
  mask2 = cv2.inRange(hsv, alt_kirmizi2, ust_kirmizi2)
  toplam_maske = cv2.bitwise_or(mask1, mask2)

  konturlar, _ = cv2.findContours(
      toplam_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  hedef_kilitte = False
  hedef_konum = None

  if len(konturlar) > 0:
    en_buyuk = max(konturlar, key=cv2.contourArea)
    if cv2.contourArea(en_buyuk) > 500:
      x, y, w, h = cv2.boundingRect(en_buyuk)
      hedef_x = x + w // 2
      hedef_y = y + h // 2
      hedef_konum = (hedef_x, hedef_y)

      hata_x = hedef_x - ekran_merkez_x
      hata_y = hedef_y - ekran_merkez_y

      # deadzone kontrolü
      if abs(hata_x) <= ESIK and abs(hata_y) <= ESIK:
        hedef_kilitte = True

      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
      cv2.line(frame, ekran_merkezi, hedef_konum, (0, 0, 255), 1)

  suan = time.time()

  # kilitlenme ve ateşlene
  if hedef_kilitte:
    if kilit_baslangic is None:
      kilit_baslangic = suan
      atis_yapildi = False

    gecen_sure = suan - kilit_baslangic

    if gecen_sure >= HEDEF_SURE:
      # ATIS GERÇEKLEŞTİ
      if not atis_yapildi:
        ses_cal(1500, 500)
        atis_yapildi = True

      # Atış anı kilit çemberi sbt yeşil halka
      if hedef_konum:
        cv2.circle(frame, hedef_konum, 20, (0, 255, 0), 2)

      cv2.putText(
          frame,
          "HEDEF VURULDU / ATIS YAPILDI",
          (50, YUKSEKLIK // 2),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.0,
          (0, 0, 255),
          3,
      )
    else:
      # geri sayımın sürmesi ve küçülen hud çemberi
      kalan_oran = (HEDEF_SURE - gecen_sure) / HEDEF_SURE
      yaricap = int(20 + 40 * kalan_oran)  # 60 pikselden 20 piksele doğru daralır

      if hedef_konum:
        cv2.circle(frame, hedef_konum, yaricap, (0, 255, 0), 2)
        cv2.drawMarker(frame, hedef_konum, (0, 255, 0), cv2.MARKER_CROSS, 10, 1)

      # ritmik geri sayım bipi çakışmayı önlemek için 1.7 saniyeye kadar
      if (suan - son_bip_zamani > 0.4) and (gecen_sure < 1.7):
        ses_cal(900, 70)
        son_bip_zamani = suan

      kalan = HEDEF_SURE - gecen_sure
      cv2.putText(
          frame,
          f"KILITLENIYOR: {kalan:.1f}s",
          (20, 35),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 255, 0),
          2,
      )

  else:
    # geri sayım başlamıştı ama atış yapılmadan hedef kaçtıysa
    if kilit_baslangic is not None and not atis_yapildi:
      ses_cal(400, 150)  # Kalın 'kilit kaybı' alarmı

    kilit_baslangic = None
    atis_yapildi = False
    cv2.putText(
        frame,
        "HEDEF ARANIYOR...",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

  # ekran merkezi nişangahı ve deadzone box
  cv2.drawMarker(frame, ekran_merkezi, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
  cv2.rectangle(
      frame,
      (ekran_merkez_x - ESIK, ekran_merkez_y - ESIK),
      (ekran_merkez_x + ESIK, ekran_merkez_y + ESIK),
      (100, 100, 100),
      1,
  )

  cv2.imshow("Kirmizi Hedef Takip ve Atis", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
