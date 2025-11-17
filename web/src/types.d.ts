// Type declarations for CDN imports

declare module "https://cdn.skypack.dev/@material/textfield" {
  export class MDCTextField {
    constructor(element: Element | null);
  }
}

declare module "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@0.10.0" {
  export class TextClassifier {
    static createFromOptions(
      resolver: any,
      options: {
        baseOptions: {
          modelAssetPath: string;
        };
        maxResults?: number;
        displayNamesLocale?: string;
      }
    ): Promise<TextClassifier>;
    classify(text: string): TextClassifierResult;
  }

  export class FilesetResolver {
    static forTextTasks(wasmPath: string): Promise<any>;
  }

  export interface TextClassifierResult {
    classifications: Array<{
      categories: Array<{
        categoryName: string;
        score: number;
      }>;
    }>;
  }
}

