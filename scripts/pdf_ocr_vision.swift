import AppKit
import Foundation
import PDFKit
import Vision

struct PageResult: Codable {
    let page: Int
    let text: String
}

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

guard CommandLine.arguments.count == 2 else {
    fail("用法：pdf_ocr_vision.swift <PDF 路径>", code: 2)
}

let path = CommandLine.arguments[1]
guard let document = PDFDocument(url: URL(fileURLWithPath: path)) else {
    fail("无法打开 PDF：\(path)")
}

var pageResults: [PageResult] = []
for pageIndex in 0..<document.pageCount {
    autoreleasepool {
        guard let page = document.page(at: pageIndex) else {
            pageResults.append(PageResult(page: pageIndex + 1, text: ""))
            return
        }

        let bounds = page.bounds(for: .mediaBox)
        let scale: CGFloat = 2.5
        let targetSize = NSSize(
            width: max(bounds.width * scale, 1),
            height: max(bounds.height * scale, 1)
        )
        let image = page.thumbnail(of: targetSize, for: .mediaBox)
        var proposedRect = NSRect(origin: .zero, size: image.size)
        guard let cgImage = image.cgImage(
            forProposedRect: &proposedRect,
            context: nil,
            hints: nil
        ) else {
            pageResults.append(PageResult(page: pageIndex + 1, text: ""))
            return
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        do {
            let supported = try request.supportedRecognitionLanguages()
            let preferred = ["zh-Hans", "zh-Hant", "en-US"].filter(supported.contains)
            if !preferred.isEmpty {
                request.recognitionLanguages = preferred
            }
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try handler.perform([request])
        } catch {
            fail("第 \(pageIndex + 1) 页 OCR 失败：\(error.localizedDescription)")
        }

        let observations = (request.results ?? []).sorted { left, right in
            let verticalDifference = abs(left.boundingBox.midY - right.boundingBox.midY)
            if verticalDifference > 0.015 {
                return left.boundingBox.midY > right.boundingBox.midY
            }
            return left.boundingBox.minX < right.boundingBox.minX
        }
        let lines = observations.compactMap { $0.topCandidates(1).first?.string }
        pageResults.append(PageResult(page: pageIndex + 1, text: lines.joined(separator: "\n")))
    }
}

do {
    let encoder = JSONEncoder()
    let data = try encoder.encode(pageResults)
    FileHandle.standardOutput.write(data)
} catch {
    fail("无法序列化 OCR 结果：\(error.localizedDescription)")
}
